from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class StreamItem:
    index: int
    text: str
    result: object

    @property
    def audio(self):
        return self.result.audio

    @property
    def sample_rate(self):
        return self.result.sample_rate


class StreamingSynthesizer:
    """Sentence-oriented delivery that makes the first completed sentence available immediately."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def stream(self, text, *, voice, speed, fame, domain=None, prefetch=False):
        sentences = self.pipeline.split_sentences(text)

        def synthesize(sentence):
            return self.pipeline.synthesize(
                sentence, voice=voice, speed=speed, fame=fame, domain=domain
            )

        if not prefetch:
            for index, sentence in enumerate(sentences, 1):
                yield StreamItem(index, sentence, synthesize(sentence))
            return
        # A single worker bounds memory and maintains model ordering.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vocaldna") as executor:
            pending = executor.submit(synthesize, sentences[0]) if sentences else None
            for index, sentence in enumerate(sentences):
                result = pending.result()
                if index + 1 < len(sentences):
                    pending = executor.submit(synthesize, sentences[index + 1])
                yield StreamItem(index + 1, sentence, result)
