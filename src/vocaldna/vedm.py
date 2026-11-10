from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class VEDMConfig:
    decay_per_second: float = 0.18
    floor: float = 0.62
    tremor_depth: float = 0.015
    tremor_hz: float = 5.5

    def __post_init__(self):
        if self.decay_per_second < 0:
            raise ValueError("decay_per_second must be non-negative")
        if not 0 <= self.floor <= 1:
            raise ValueError("floor must be between 0 and 1")
        if self.tremor_depth < 0:
            raise ValueError("tremor_depth must be non-negative")


class VEDM:
    """Deterministic voice-energy shaping; breath synthesis is intentionally external."""

    def __init__(self, config: VEDMConfig | None = None):
        self.config = config or VEDMConfig()
        self.energy = 1.0

    def energy_curve(self, sample_count: int, sample_rate: int, *, initial_energy: float = 1.0) -> np.ndarray:
        if sample_count <= 0:
            return np.empty(0, dtype=np.float32)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        time = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
        config = self.config
        curve = config.floor + (initial_energy - config.floor) * np.exp(-config.decay_per_second * time)
        curve = np.maximum(curve, config.floor)
        if config.tremor_depth:
            low_energy = 1.0 - curve
            curve *= 1.0 + config.tremor_depth * low_energy * np.sin(2.0 * np.pi * config.tremor_hz * time)
        curve = np.maximum(curve, config.floor)
        return curve.astype(np.float32)

    def apply(self, audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        samples = np.asarray(audio, dtype=np.float32)
        sample_count = samples.shape[0] if samples.ndim else samples.size
        curve = self.energy_curve(sample_count, sample_rate)
        if samples.ndim > 1:
            curve = curve.reshape((sample_count,) + (1,) * (samples.ndim - 1))
        return samples * curve

    def process(self, audio: np.ndarray, *, sample_rate: int, pause_after: float = 0.0) -> np.ndarray:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        samples = np.asarray(audio, dtype=np.float32)
        sample_count = samples.shape[0] if samples.ndim else samples.size
        duration = sample_count / float(sample_rate)
        initial_energy = max(self.config.floor, self.energy)
        curve = self.energy_curve(sample_count, sample_rate, initial_energy=initial_energy)
        if samples.ndim > 1:
            curve = curve.reshape((sample_count,) + (1,) * (samples.ndim - 1))
        result = samples * curve
        self.energy = max(
            self.config.floor,
            initial_energy * np.exp(-self.config.decay_per_second * duration),
        )
        if pause_after > 0:
            recovery = 1.0 - np.exp(-0.7 * float(pause_after))
            self.energy += (1.0 - self.energy) * recovery
        return result.astype(np.float32)

    def describe(self, duration_seconds: float) -> dict[str, object]:
        return {
            "model": "VEDM",
            "durationSeconds": float(duration_seconds),
            "config": asdict(self.config),
            "energyAfter": self.energy,
            "breathInsertion": "external",
        }
