import json
import tempfile
import unittest
from pathlib import Path

from vocaldna.g2p import G2PTable


class G2PTableTests(unittest.TestCase):
    def setUp(self):
        self.entries = {
            "会计": {
                "pinyin": "kuài jì",
                "domain": "财务",
                "pos": "名词",
            },
            "会": {"pinyin": "huì", "domain": "日常"},
            "计": {"pinyin": "jì", "domain": "日常"},
            "你好": {"pinyin": "nǐ hǎo", "domain": "日常"},
            "不对": {"pinyin": "bù duì", "domain": "日常"},
            "一共": {"pinyin": "yī gòng", "domain": "日常"},
        }

    def test_longest_matching_uses_word_pronunciation(self):
        table = G2PTable(self.entries)

        tokens = table.convert("会计")

        self.assertEqual([token.surface for token in tokens], ["会计"])
        self.assertEqual(tokens[0].pinyin, ["kuai4", "ji4"])
        self.assertEqual(tokens[0].metadata["domain"], "财务")

    def test_domain_filter_falls_back_to_character_entries(self):
        table = G2PTable(self.entries)

        tokens = table.convert("会计", domain="日常")

        self.assertEqual([token.surface for token in tokens], ["会", "计"])
        self.assertEqual([token.pinyin for token in tokens], [["hui4"], ["ji4"]])

    def test_punctuation_is_preserved_as_a_boundary(self):
        table = G2PTable(self.entries)

        tokens = table.convert("你好！")

        self.assertEqual([token.kind for token in tokens], ["word", "punctuation"])
        self.assertEqual(tokens[-1].surface, "！")
        self.assertEqual(tokens[-1].pinyin, [])

    def test_override_has_priority_over_lexicon(self):
        table = G2PTable(self.entries, overrides={"会计": "kuài jǐ"})

        tokens = table.convert("会计")

        self.assertEqual(tokens[0].pinyin, ["kuai4", "ji3"])
        self.assertTrue(tokens[0].metadata["override"])

    def test_unknown_chinese_character_uses_fallback(self):
        table = G2PTable({})

        tokens = table.convert("喵")

        self.assertEqual(tokens[0].kind, "fallback")
        self.assertEqual(tokens[0].pinyin, ["miao1"])

    def test_json_resource_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lexicon.json"
            path.write_text(json.dumps(self.entries, ensure_ascii=False), encoding="utf-8")

            table = G2PTable.from_json(path)

            self.assertEqual(table.lookup("会计").pinyin, ["kuai4", "ji4"])

    def test_third_tone_sandhi_and_function_word_rules(self):
        table = G2PTable(self.entries)

        self.assertEqual(table.pronounce("你好"), ["ni2", "hao3"])
        self.assertEqual(table.pronounce("不对"), ["bu2", "dui4"])
        self.assertEqual(table.pronounce("一共"), ["yi2", "gong4"])

    def test_tone_sandhi_does_not_cross_sentence_boundaries(self):
        table = G2PTable({
            "你": {"pinyin": "nǐ"},
            "好": {"pinyin": "hǎo"},
        })

        self.assertEqual(table.pronounce("你，好"), ["ni3", "hao3"])


if __name__ == "__main__":
    unittest.main()
