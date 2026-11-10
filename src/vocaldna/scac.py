from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class AlignmentResult:
    passed: bool
    mismatches: list[tuple[int, str, str | None]]
    attempts: int = 1
    available: bool = True


def validate_alignment(expected: Sequence[str], observed: Sequence[str]) -> AlignmentResult:
    expected = list(expected)
    observed = list(observed)
    mismatches = []
    for index in range(max(len(expected), len(observed))):
        left = expected[index] if index < len(expected) else None
        right = observed[index] if index < len(observed) else None
        if left != right:
            mismatches.append((index, left or "", right))
    return AlignmentResult(not mismatches, mismatches)


class SCACController:
    """Optional pronunciation feedback loop for engines that expose observed phonemes."""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        aligner: Callable | None = None,
        retry_policy: Callable[[int, Sequence[tuple[int, str, str | None]]], Mapping] | None = None,
    ):
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.max_retries = max_retries
        self.aligner = aligner
        self.retry_policy = retry_policy

    def run(
        self,
        *,
        expected: Sequence[str],
        synthesize: Callable[[int], Iterable[str]],
    ) -> AlignmentResult:
        last = AlignmentResult(False, [], 0)
        for attempt in range(1, self.max_retries + 1):
            last = validate_alignment(expected, list(synthesize(attempt)))
            if last.passed:
                return AlignmentResult(True, [], attempt)
        return AlignmentResult(False, last.mismatches, self.max_retries)

    def verify_audio(self, *, expected, audio, sample_rate, synthesize):
        if self.aligner is None:
            return audio, AlignmentResult(False, [], 0, available=False)
        candidate = audio
        for attempt in range(1, self.max_retries + 1):
            samples = candidate.audio if hasattr(candidate, "audio") else candidate
            observed = self.aligner(samples, sample_rate)
            result = validate_alignment(expected, list(observed))
            if result.passed:
                return candidate, AlignmentResult(True, [], attempt)
            if attempt < self.max_retries:
                corrections = (
                    self.retry_policy(attempt + 1, result.mismatches)
                    if self.retry_policy
                    else {}
                )
                candidate = synthesize(attempt + 1, corrections)
        return candidate, AlignmentResult(
            False,
            result.mismatches,
            self.max_retries,
        )
