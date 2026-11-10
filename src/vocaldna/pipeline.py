from __future__ import annotations

import re
import inspect
import math
import threading
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .emotion import FAME
from .engine import AudioResult, SynthesisEngine
from .g2p import G2PTable
from .normalize import TextNormalizer
from .scac import SCACController
from .streaming import StreamingSynthesizer
from .vedm import VEDM


@dataclass
class SynthesisOutput:
    audio: np.ndarray
    sample_rate: int
    voice: str
    text: str
    normalized_text: str
    metadata: dict[str, object] = field(default_factory=dict)


class VocalDNA:
    """The complete table-first pipeline with replaceable synthesis engines."""

    def __init__(
        self,
        *,
        g2p: G2PTable,
        engine: SynthesisEngine,
        normalizer: TextNormalizer | None = None,
        vedm: VEDM | None = None,
        scac: SCACController | None = None,
    ):
        self.g2p = g2p
        self.engine = engine
        self.normalizer = normalizer or TextNormalizer()
        self.vedm = vedm or VEDM()
        self.scac = scac or SCACController()
        self._lock = threading.RLock()

    @staticmethod
    def split_sentences(text: str, max_chars: int = 100) -> list[str]:
        text = re.sub(r"[^\S\n]+", " ", str(text or "").strip())
        if not text:
            return []
        parts = []
        for part in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text):
            part = part.strip()
            while part:
                parts.append(part[:max_chars])
                part = part[max_chars:]
        return parts

    def synthesize(
        self, text, *, voice="zf_001", speed=1.0, fame=None, domain=None
    ) -> SynthesisOutput:
        with self._lock:
            return self._synthesize(text, voice=voice, speed=speed, fame=fame, domain=domain)

    def _synthesize(
        self,
        text: str,
        *,
        voice: str = "zf_001",
        speed: float = 1.0,
        fame: FAME | Mapping[str, float] | None = None,
        domain: str | list[str] | None = None,
    ) -> SynthesisOutput:
        normalized = self.normalizer.normalize(text)
        if not normalized or not any(char.isalnum() for char in normalized):
            raise ValueError("text must contain a pronounceable character")
        if not math.isfinite(float(speed)) or not 0.25 <= float(speed) <= 4.0:
            raise ValueError("speed must be finite and between 0.25 and 4")
        fame_value = fame if isinstance(fame, FAME) else FAME.from_mapping(fame)
        tokens = self.g2p.apply_sandhi(self.g2p.convert(normalized, domain=domain))
        pronunciation = [p for tk in tokens for p in tk.pinyin]
        raw = self._synthesize_engine(
            normalized,
            voice=voice,
            speed=float(speed),
            fame=fame_value,
            pronunciation=pronunciation,
            pronunciation_tokens=tokens,
        )
        scac_status = {"status": "not_configured", "passed": None, "attempts": 0}
        if self.scac.aligner is not None:
            raw, checked = self.scac.verify_audio(
                expected=pronunciation,
                audio=raw,
                sample_rate=raw.sample_rate,
                synthesize=lambda _attempt, corrections: self._synthesize_engine(
                    normalized,
                    voice=voice,
                    speed=float(corrections.get("speed", speed)),
                    fame=corrections.get("fame", fame_value),
                    pronunciation=pronunciation,
                    pronunciation_tokens=tokens,
                ),
            )
            scac_status = {
                "status": "verified",
                "passed": checked.passed,
                "attempts": checked.attempts,
                "mismatches": checked.mismatches,
            }
        raw_audio = np.array(raw.audio, dtype=np.float32, copy=True)
        if raw_audio.ndim != 1 or raw_audio.size == 0 or not np.isfinite(raw_audio).all():
            raise ValueError("engine audio must be finite nonempty mono samples")
        if not isinstance(raw.sample_rate, int) or raw.sample_rate <= 0:
            raise ValueError("engine sample rate must be a positive integer")
        raw_audio *= 10 ** (fame_value.to_prosody().volume_db / 20.0)
        duration = raw_audio.shape[0] / raw.sample_rate
        pause_after = 0.16 * fame_value.to_prosody().pause_scale
        if normalized.endswith(("。", "！", "？", "!", "?", "；", ";")):
            pause_after *= 1.5
        processed = self.vedm.process(
            raw_audio,
            sample_rate=raw.sample_rate,
            pause_after=pause_after,
        )
        metadata = dict(raw.metadata)
        metadata.update({
            "tokens": [
                {"surface": token.surface, "pinyin": token.pinyin, "kind": token.kind, **token.metadata}
                for token in tokens
            ],
            "pronunciation": pronunciation,
            "fame": fame_value.to_vector(),
            "prosody": fame_value.to_prosody().__dict__,
            "vedm": self.vedm.describe(duration),
            "scac": scac_status,
        })
        return SynthesisOutput(
            audio=np.asarray(processed, dtype=np.float32),
            sample_rate=raw.sample_rate,
            voice=raw.voice,
            text=str(text),
            normalized_text=normalized,
            metadata=metadata,
        )

    def _synthesize_engine(self, text, **kwargs):
        signature = inspect.signature(self.engine.synthesize)
        if "pronunciation_tokens" not in signature.parameters:
            kwargs.pop("pronunciation_tokens", None)
        return self.engine.synthesize(text, **kwargs)

    def synthesize_stream(self, text, *, voice="zf_001", speed=1.0, fame=None, domain=None, prefetch=False):
        return StreamingSynthesizer(self).stream(
            text,
            voice=voice,
            speed=speed,
            fame=fame,
            domain=domain,
            prefetch=prefetch,
        )
