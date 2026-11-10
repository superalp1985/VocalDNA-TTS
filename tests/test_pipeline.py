import unittest
import importlib.util

import numpy as np

from vocaldna.engine import KokoroEngine
from vocaldna.engine import AudioResult, MemoryEngine
from vocaldna.scac import SCACController
from vocaldna.g2p import G2PTable, PronunciationToken
from vocaldna.pipeline import VocalDNA


class PipelineTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("jieba"), "install the Kokoro extra")
    def test_kokoro_adapter_registers_lexicon_readings_with_its_g2p(self):
        engine = KokoroEngine()

        engine._install_lexicon_overrides([
            PronunciationToken("会计", ["kuai4", "ji4"], metadata={"matched": True})
        ])

        from pypinyin import Style, lazy_pinyin
        self.assertEqual(lazy_pinyin("会计", style=Style.TONE3), ["kuai4", "ji4"])

    def test_stream_yields_first_sentence_before_later_sentences_finish(self):
        events = []

        class Engine:
            sample_rate = 24000

            def synthesize(self, text, *, voice, speed, fame, pronunciation):
                events.append(text)
                return AudioResult(np.ones(8, dtype=np.float32), self.sample_rate, voice)

        g2p = G2PTable({"第一句": {"pinyin": "dì yī jù"}, "第二句": {"pinyin": "dì èr jù"}})
        pipeline = VocalDNA(g2p=g2p, engine=Engine())
        stream = pipeline.synthesize_stream("第一句。第二句。", voice="zf_001")

        first = next(stream)

        self.assertEqual(first.index, 1)
        self.assertEqual(first.text, "第一句。")
        self.assertEqual(events, ["第一句。"])
        self.assertEqual(next(stream).index, 2)

    def test_pipeline_applies_fame_and_vedm_to_engine_audio(self):
        pipeline = VocalDNA(
            g2p=G2PTable({"你好": {"pinyin": "nǐ hǎo"}}),
            engine=MemoryEngine(sample_rate=24000),
        )

        result = pipeline.synthesize("你好。", voice="zf_001")

        self.assertEqual(result.sample_rate, 24000)
        self.assertEqual(result.voice, "zf_001")
        self.assertTrue(result.metadata["pronunciation"])

    def test_pipeline_retries_after_scac_alignment_mismatch(self):
        calls = []

        class Engine:
            sample_rate = 24000

            def synthesize(self, text, **kwargs):
                calls.append(text)
                return AudioResult(np.ones(100, dtype=np.float32), self.sample_rate, kwargs["voice"])

        alignments = iter([["wrong"], ["ni2", "hao3"]])
        pipeline = VocalDNA(
            g2p=G2PTable({"你好": {"pinyin": "nǐ hǎo"}}),
            engine=Engine(),
            scac=SCACController(aligner=lambda audio, rate: next(alignments)),
        )

        result = pipeline.synthesize("你好")

        self.assertEqual(len(calls), 2)
        self.assertTrue(result.metadata["scac"]["passed"])
        self.assertEqual(result.metadata["scac"]["attempts"], 2)


if __name__ == "__main__":
    unittest.main()
