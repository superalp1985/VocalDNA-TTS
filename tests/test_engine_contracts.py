import unittest
import importlib.util
from types import SimpleNamespace

from vocaldna.bert import BertVITS2Engine
from vocaldna.emotion import FAME
from vocaldna.g2p import PronunciationToken
from vocaldna.phonemes import kokoro_phone_units
from vocaldna.bert import make_cleaner, build_emotion_prompt


class EngineContractTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("misaki"), "install the Kokoro extra")
    def test_kokoro_phonemes_use_exact_table_tones(self):
        tokens = [
            PronunciationToken("会计", ["kuai4", "ji4"]),
            PronunciationToken("。", [], "punctuation"),
        ]
        units = kokoro_phone_units(tokens)
        self.assertEqual(units, ["ㄎ外4", "ㄐㄧ4", "."])

    def test_bert_cleaner_uses_table_phones_and_keeps_bert_alignment(self):
        chinese = SimpleNamespace(
            text_normalize=lambda text: text.replace("。", "."),
            pinyin_to_symbol_map={"kuai": "k uai", "ji": "j i"},
        )
        cleaner = make_cleaner([
            PronunciationToken("会计", ["kuai4", "ji4"]),
            PronunciationToken("。", [], "punctuation"),
        ], chinese)

        text, phones, tones, word2ph = cleaner("会计。", "ZH")

        self.assertEqual(text, "会计.")
        self.assertEqual(phones, ["_", "k", "uai", "j", "i", ".", "_"])
        self.assertEqual(tones, [0, 4, 4, 4, 4, 0, 0])
        self.assertEqual(word2ph, [1, 2, 2, 1, 1])
        self.assertEqual(sum(word2ph), len(phones))
        self.assertEqual(len(word2ph), len(text) + 2)

    def test_emotion_prompt_represents_all_eight_dimensions(self):
        prompt = build_emotion_prompt(FAME(joy=0.8, anger=0.6, tenderness=0.3))
        self.assertIn("joy", prompt)
        self.assertIn("anger", prompt)
        self.assertIn("tenderness", prompt)

    def test_bert_adapter_passes_table_readings_and_clap_condition(self):
        received = {}

        def infer(**kwargs):
            received.update(kwargs)
            return [0.1, -0.1]

        engine = BertVITS2Engine(
            infer,
            condition=lambda prompt: ("embedding", prompt),
            sample_rate=44100,
        )
        tokens = [PronunciationToken("会计", ["kuai4", "ji4"])]

        result = engine.synthesize(
            "会计",
            voice="speaker-1",
            speed=1.0,
            fame=FAME(joy=0.8),
            pronunciation=["kuai4", "ji4"],
            pronunciation_tokens=tokens,
        )

        self.assertEqual(result.sample_rate, 44100)
        self.assertEqual(received["pronunciation"], ["kuai4", "ji4"])
        self.assertIsNotNone(received["emotion_embedding"])
        self.assertEqual(len(received["fame_vector"]), 8)
        self.assertIsNone(received["text_cleaner"])


if __name__ == "__main__":
    unittest.main()
