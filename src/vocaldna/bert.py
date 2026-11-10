from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .engine import AudioResult
from .g2p import PronunciationToken
from .phonemes import split_syllable


def build_emotion_prompt(fame) -> str:
    values = {
        "joy": fame.joy,
        "calm": fame.calm,
        "anger": fame.anger,
        "sadness": fame.sadness,
        "confidence": fame.confidence,
        "formality": fame.formality,
        "surprise": fame.surprise,
        "tenderness": fame.tenderness,
    }
    return "VocalDNA FAME " + " ".join(f"{key}={value:.3f}" for key, value in values.items())


def make_cleaner(tokens: Iterable[PronunciationToken], chinese) -> Callable:
    """Create a Bert-VITS2-compatible cleaner using table phonemes and tones."""
    table_tokens = list(tokens)
    symbol_map = chinese.pinyin_to_symbol_map

    def clean(text: str, language: str):
        normalized = chinese.text_normalize(text)
        phones = ["_"]
        tones = [0]
        word2ph = [1]
        for token in table_tokens:
            if token.kind == "punctuation":
                punctuation = chinese.text_normalize(token.surface)
                for char in punctuation:
                    phones.append(char)
                    tones.append(0)
                    word2ph.append(1)
                continue
            if len(token.surface) != len(token.pinyin):
                raise ValueError(f"table entry does not align to characters: {token.surface}")
            for character, syllable in zip(token.surface, token.pinyin):
                initial, final, tone = split_syllable(syllable)
                base = initial + final
                if base not in symbol_map:
                    raise ValueError(f"Bert-VITS2 has no phoneme mapping for {syllable}")
                phones_for_char = symbol_map[base].split()
                phones.extend(phones_for_char)
                tones.extend([tone] * len(phones_for_char))
                word2ph.append(len(phones_for_char))
        source_text = chinese.text_normalize("".join(token.surface for token in table_tokens))
        if source_text != normalized:
            raise ValueError("table tokens do not match Bert-VITS2 normalized text")
        phones.append("_")
        tones.append(0)
        word2ph.append(1)
        if sum(word2ph) != len(phones) or len(word2ph) != len(normalized) + 2:
            raise ValueError("table G2P and Bert-VITS2 normalized text are misaligned")
        return normalized, phones, tones, word2ph

    return clean


class BertVITS2Engine:
    """Adapter for an externally supplied Bert-VITS2 inference callback."""

    def __init__(
        self,
        infer: Callable,
        *,
        sample_rate: int = 44100,
        condition: Callable | None = None,
        chinese_frontend=None,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.infer = infer
        self.condition = condition
        self.chinese_frontend = chinese_frontend
        self.sample_rate = int(sample_rate)

    def synthesize(self, text, *, voice, speed, fame, pronunciation, pronunciation_tokens):
        prompt = build_emotion_prompt(fame)
        emotion_embedding = self.condition(prompt) if self.condition else None
        text_cleaner = (
            make_cleaner(pronunciation_tokens, self.chinese_frontend)
            if self.chinese_frontend is not None
            else None
        )
        audio = self.infer(
            text=text,
            voice=voice,
            speed=float(speed) * fame.to_prosody().speed,
            emotion_prompt=prompt,
            emotion_embedding=emotion_embedding,
            fame_vector=fame.to_vector(),
            prosody=fame.to_prosody(),
            pronunciation=list(pronunciation),
            pronunciation_tokens=list(pronunciation_tokens),
            text_cleaner=text_cleaner,
        )
        if isinstance(audio, AudioResult):
            return audio
        return AudioResult(
            np.asarray(audio, dtype=np.float32),
            self.sample_rate,
            voice,
            {
                "engine": "bert-vits2-extra",
                "clapConditioned": emotion_embedding is not None,
                "emotionPrompt": prompt,
                "pronunciationRequested": list(pronunciation),
            },
        )
