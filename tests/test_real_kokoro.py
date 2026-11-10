import os
import unittest
from pathlib import Path

import numpy as np

from vocaldna import G2PTable, KokoroEngine, VocalDNA


class RealKokoroTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("VOCALDNA_TEST_MODEL_DIR"), "set VOCALDNA_TEST_MODEL_DIR to run local-model integration")
    def test_local_model_generates_nonempty_dictionary_conditioned_audio(self):
        model_dir = Path(os.environ["VOCALDNA_TEST_MODEL_DIR"])
        pipeline = VocalDNA(
            g2p=G2PTable.from_resource(),
            engine=KokoroEngine(model_dir),
        )

        output = pipeline.synthesize("会计。")

        self.assertEqual(output.sample_rate, 24000)
        self.assertEqual(output.voice, "zf_001")
        self.assertGreater(len(output.audio), 2400)
        self.assertTrue(np.isfinite(output.audio).all())
        self.assertEqual(output.metadata["tokens"][0]["pinyin"], ["kuai4", "ji4"])
        self.assertEqual(output.metadata["scac"]["status"], "not_configured")


if __name__ == "__main__":
    unittest.main()
