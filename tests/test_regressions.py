import unittest
import importlib.util
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from vocaldna.engine import AudioResult
from vocaldna.g2p import G2PTable
from vocaldna.normalize import TextNormalizer
from vocaldna.phonemes import kokoro_phone_units
from vocaldna.pipeline import VocalDNA


class RegressionTests(unittest.TestCase):
    def test_four_digit_groups_include_big_units_when_group_ends_in_zero(self):
        normalizer = TextNormalizer()
        for number, expected in {
            "10000": "一万",
            "10010": "一万零一十",
            "100000000": "一亿",
            "100010000": "一亿零一万",
            "1001": "一千零一",
        }.items():
            with self.subTest(number=number):
                self.assertEqual(normalizer.normalize(number), expected)

    def test_sandhi_applies_only_to_function_characters(self):
        table = G2PTable({
            "布料": {"pinyin": "bù liào"},
            "医院": {"pinyin": "yī yuàn"},
            "第一": {"pinyin": "dì yī"},
            "一百": {"pinyin": "yī bǎi"},
        })
        self.assertEqual(table.pronounce("布料"), ["bu4", "liao4"])
        self.assertEqual(table.pronounce("医院"), ["yi1", "yuan4"])
        self.assertEqual(table.pronounce("第一百"), ["di4", "yi1", "bai3"])
        self.assertEqual(table.pronounce("一百"), ["yi4", "bai3"])

    @unittest.skipUnless(importlib.util.find_spec("misaki"), "install the Kokoro extra")
    def test_neutral_tone_and_umlaut_have_valid_kokoro_tokens(self):
        table = G2PTable({
            "女儿": {"pinyin": "nǚ ér"},
            "是": {"pinyin": "shì"},
            "好": {"pinyin": "hǎo"},
            "的": {"pinyin": "de"},
        })
        self.assertEqual(table.lookup("的").pinyin, ["de5"])
        self.assertEqual(
            kokoro_phone_units(table.convert("女儿是好的")),
            ["ㄋㄩ3", "ㄦ2", "ㄕ十4", "ㄏㄠ3", "ㄉㄜ5"],
        )

    def test_prefetch_prepares_next_sentence_during_consumer_pause(self):
        import threading
        started = threading.Event()

        class Engine:
            sample_rate = 24000

            def synthesize(self, text, **kwargs):
                if text.startswith("第二"):
                    started.set()
                return AudioResult(np.zeros(50, np.float32), self.sample_rate, kwargs["voice"])

        pipeline = VocalDNA(g2p=G2PTable({}), engine=Engine())
        stream = pipeline.synthesize_stream("第一句。第二句。", prefetch=True)
        first = next(stream)
        self.assertEqual(first.index, 1)
        self.assertTrue(started.wait(2))
        self.assertEqual(next(stream).index, 2)
        stream.close()

    def test_model_inputs_and_scac_are_finite_and_failures_are_visible(self):
        class Engine:
            sample_rate = 24000

            def synthesize(self, text, **kwargs):
                return AudioResult(np.array([float("nan")]), 24000, kwargs["voice"])

        pipeline = VocalDNA(g2p=G2PTable({}), engine=Engine())
        with self.assertRaises(ValueError):
            pipeline.synthesize("测试")

    def test_sentence_split_preserves_newline_and_bounds_long_sentences(self):
        text = "甲\n乙。" + "丙" * 401
        parts = VocalDNA.split_sentences(text)
        self.assertEqual("".join(parts), text.replace("\n", ""))
        self.assertTrue(all(len(part) <= 100 for part in parts))


if __name__ == "__main__":
    unittest.main()
