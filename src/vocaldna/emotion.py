from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping


def _bounded(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class Prosody:
    speed: float
    pitch_semitones: float
    volume_db: float
    pause_scale: float
    emphasis: float


@dataclass(frozen=True)
class FAME:
    """Eight bounded controls used by engines to derive delivery parameters."""

    joy: float = 0.0
    calm: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0
    confidence: float = 0.0
    formality: float = 0.0
    surprise: float = 0.0
    tenderness: float = 0.0

    def __post_init__(self):
        for item in fields(self):
            object.__setattr__(self, item.name, _bounded(getattr(self, item.name), item.name))

    @classmethod
    def from_mapping(cls, values: Mapping[str, float] | None = None) -> "FAME":
        if not values:
            return cls()
        allowed = {item.name for item in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown FAME dimensions: {sorted(unknown)}")
        return cls(**{name: values[name] for name in values})

    def to_vector(self) -> list[float]:
        return [getattr(self, item.name) for item in fields(self)]

    def to_prosody(self) -> Prosody:
        speed = (
            1.0
            + 0.20 * self.joy
            + 0.12 * self.anger
            + 0.10 * self.surprise
            + 0.05 * self.confidence
            - 0.16 * self.calm
            - 0.22 * self.sadness
            - 0.10 * self.tenderness
        )
        pitch = (
            0.9 * self.joy
            + 0.35 * self.anger
            + 1.2 * self.surprise
            - 1.1 * self.sadness
            - 0.25 * self.tenderness
        )
        volume = 2.0 * self.anger + 0.5 * self.confidence - self.sadness - 0.5 * self.tenderness
        pause_scale = 1.0 + 0.20 * self.calm + 0.20 * self.sadness + 0.10 * self.tenderness - 0.10 * self.anger
        emphasis = min(1.0, 0.45 * self.confidence + 0.30 * self.anger + 0.25 * self.surprise)
        return Prosody(
            speed=max(0.55, min(1.65, speed)),
            pitch_semitones=max(-3.0, min(3.0, pitch)),
            volume_db=max(-6.0, min(6.0, volume)),
            pause_scale=max(0.65, min(1.8, pause_scale)),
            emphasis=emphasis,
        )
