import unittest

from vocaldna.g2p import G2PTable


class BundledLexiconTests(unittest.TestCase):
    def test_bundled_table_has_all_published_entries_and_expected_reading(self):
        table = G2PTable.from_resource()

        self.assertEqual(len(table), 578895)
        entry = table.lookup("会计")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.pinyin, ["kuai4", "ji4"])


if __name__ == "__main__":
    unittest.main()
