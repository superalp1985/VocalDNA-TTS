from __future__ import annotations

import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .emotion import FAME
from .g2p import PronunciationToken


@dataclass
class AudioResult:
    audio: np.ndarray
    sample_rate: int
    voice: str
    metadata: dict[str, object] = field(default_factory=dict)


class SynthesisEngine(Protocol):
    def synthesize(
        self,
        text: str,
        *,
        voice: str,
        speed: float,
        fame: FAME,
        pronunciation: Sequence[str],
        pronunciation_tokens: Sequence[PronunciationToken],
    ) -> AudioResult:
        ...


class MemoryEngine:
    """Small deterministic engine for tests and pipeline integration examples."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def synthesize(self, text, *, voice, speed, fame, pronunciation, pronunciation_tokens):
        duration = max(1, len(text)) * 32
        return AudioResult(
            np.ones(duration, dtype=np.float32),
            self.sample_rate,
            voice,
            {"pronunciationRequested": list(pronunciation), "engine": "memory"},
        )


class _NullStream:
    def write(self, value):
        return len(value or "")

    def flush(self):
        return None

    def isatty(self):
        return False


def _ensure_streams():
    if getattr(sys, "stderr", None) is None:
        sys.stderr = _NullStream()
    if getattr(sys, "stdout", None) is None:
        sys.stdout = _NullStream()


class KokoroEngine:
    """Optional local Kokoro adapter. Model files stay outside this repository."""

    sample_rate = 24000
    default_voice = "zf_001"
    repo_id = "hexgrad/Kokoro-82M-v1.1-zh"

    def __init__(self, model_dir: str | Path | None = None, *, device: str | None = None):
        configured = os.environ.get("VOCALDNA_KOKORO_MODEL_DIR", "").strip()
        self.model_dir = Path(model_dir or configured) if (model_dir or configured) else None
        self.device = device
        self.model = None
        self.pipeline = None

    def _require_model_dir(self) -> Path:
        if self.model_dir is None:
            raise FileNotFoundError(
                "Kokoro model directory is not configured. Set VOCALDNA_KOKORO_MODEL_DIR."
            )
        config = self.model_dir / "config.json"
        weights = self.model_dir / "kokoro-v1_1-zh.pth"
        if not config.is_file() or not weights.is_file():
            raise FileNotFoundError(f"incomplete Kokoro model directory: {self.model_dir}")
        return self.model_dir

    def init(self):
        if self.pipeline is not None:
            return
        _ensure_streams()
        model_dir = self._require_model_dir()
        import torch
        from kokoro import KModel, KPipeline

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KModel(
            repo_id=self.repo_id,
            config=str(model_dir / "config.json"),
            model=str(model_dir / "kokoro-v1_1-zh.pth"),
        ).to(device).eval()
        self.pipeline = KPipeline(lang_code="z", repo_id=self.repo_id, model=self.model)

    def list_voices(self) -> list[str]:
        if self.model_dir is None:
            return []
        voice_dir = self.model_dir / "voices"
        return sorted(path.stem for path in voice_dir.glob("*.pt")) if voice_dir.is_dir() else []

    @staticmethod
    def _install_lexicon_overrides(tokens: Sequence[PronunciationToken]):
        import jieba
        from pypinyin import load_phrases_dict

        phrases = {}
        for token in tokens:
            if token.kind != "word" or len(token.surface) != len(token.pinyin):
                continue
            if not all("\u3400" <= char <= "\u9fff" for char in token.surface):
                continue
            if not all(item and item[-1:] in "12345" for item in token.pinyin):
                continue
            phrases[token.surface] = [[item] for item in token.pinyin]
            jieba.add_word(token.surface, freq=10**9)
        if phrases:
            load_phrases_dict(phrases)

    def synthesize(self, text, *, voice, speed, fame, pronunciation, pronunciation_tokens):
        self.init()
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", voice) or voice in {".", ".."}:
            raise ValueError("voice contains unsupported characters")
        voice_file = self.model_dir / "voices" / f"{voice}.pt"
        if not voice_file.is_file():
            raise FileNotFoundError(f"Kokoro voice is not installed: {voice}")
        voice_arg = str(voice_file)
        actual_speed = max(0.55, min(1.65, float(speed) * fame.to_prosody().speed))
        with _LEXICON_LOCK:
            self._install_lexicon_overrides(pronunciation_tokens)
            return self._synthesize_locked(
                text,
                voice=voice,
                voice_arg=voice_arg,
                speed=actual_speed,
                pronunciation=pronunciation,
                pronunciation_tokens=pronunciation_tokens,
            )

    def _synthesize_locked(self, text, *, voice, voice_arg, speed, pronunciation, pronunciation_tokens):
        chunks = []
        for result in self.pipeline(text, voice=voice_arg, speed=speed):
            audio = result.audio
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            if audio is not None and len(audio):
                chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Kokoro returned no audio")
        return AudioResult(
            np.concatenate(chunks),
            self.sample_rate,
            voice,
            {
                "engine": "kokoro",
                "pronunciationRequested": list(pronunciation),
                "lexiconEntriesInjected": sum(token.kind == "word" for token in pronunciation_tokens),
                "speed": speed,
            },
        )


_LEXICON_LOCK = threading.RLock()
