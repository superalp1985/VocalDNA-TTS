import unittest

from vocaldna.normalize import TextNormalizer


class TextNormalizerTests(unittest.TestCase):
    def test_normalizes_numbers_units_symbols_and_abbreviations(self):
        normalizer = TextNormalizer()

        result = normalizer.normalize("ASR 50% + 2kg，API v2.0！")

        self.assertEqual(result, "诶艾斯阿尔百分之五十加二千克，诶批爱维二点零！")

    def test_preserves_chinese_text_and_removes_layout_whitespace(self):
        normalizer = TextNormalizer()

        self.assertEqual(normalizer.normalize("  你好  世界 "), "你好世界")


if __name__ == "__main__":
    unittest.main()
