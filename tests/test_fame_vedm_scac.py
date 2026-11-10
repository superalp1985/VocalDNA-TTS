import unittest

import numpy as np

from vocaldna.emotion import FAME
from vocaldna.scac import SCACController, validate_alignment
from vocaldna.vedm import VEDM, VEDMConfig


class FAMEVEDMSCACTests(unittest.TestCase):
    def test_fame_requires_eight_bounded_dimensions_and_maps_to_prosody(self):
        fame = FAME(joy=0.8, calm=0.2, anger=0.1, sadness=0.0,
                    confidence=0.7, formality=0.6, surprise=0.3, tenderness=0.4)

        prosody = fame.to_prosody()

        self.assertEqual(len(fame.to_vector()), 8)
        self.assertGreater(prosody.speed, 1.0)
        self.assertGreater(prosody.pitch_semitones, 0.0)
        with self.assertRaises(ValueError):
            FAME(joy=1.2)

    def test_vedm_applies_deterministic_energy_decay(self):
        audio = np.ones(48000, dtype=np.float32)
        vedm = VEDM(VEDMConfig(decay_per_second=0.5, floor=0.2, tremor_depth=0.0))

        first = vedm.apply(audio, sample_rate=24000)
        second = vedm.apply(audio, sample_rate=24000)

        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(first[:100].mean()), float(first[-100:].mean()))
        self.assertGreaterEqual(float(first[-1]), 0.2)

    def test_vedm_state_recovers_during_pause_and_keeps_stereo_shape(self):
        vedm = VEDM(VEDMConfig(decay_per_second=0.7, floor=0.2, tremor_depth=0.0))
        stereo = np.ones((24000, 2), dtype=np.float32)

        first = vedm.process(stereo, sample_rate=24000)
        depleted = vedm.energy
        second = vedm.process(stereo, sample_rate=24000, pause_after=1.0)

        self.assertEqual(first.shape, stereo.shape)
        self.assertEqual(second.shape, stereo.shape)
        self.assertGreater(vedm.energy, depleted)
        self.assertLess(float(second[:, 0].mean()), float(first[:, 0].mean()))

    def test_scac_reports_mismatch_and_retries(self):
        controller = SCACController(max_retries=2)
        calls = []

        def synthesize(attempt):
            calls.append(attempt)
            return ["ni3", "hao3"] if attempt == 2 else ["ni2", "hao3"]

        result = controller.run(
            expected=["ni3", "hao3"],
            synthesize=synthesize,
        )

        self.assertTrue(result.passed)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(validate_alignment(["a"], ["b"]).mismatches, [(0, "a", "b")])

    def test_scac_audio_verifier_retries_with_regenerated_audio(self):
        corrections_seen = []
        controller = SCACController(
            max_retries=2,
            aligner=lambda audio, sample_rate: audio,
            retry_policy=lambda attempt, mismatches: (
                corrections_seen.append((attempt, mismatches)) or {"reading": "right"}
            ),
        )
        outputs = []

        audio, result = controller.verify_audio(
            expected=["right"],
            audio=["wrong"],
            sample_rate=24000,
            synthesize=lambda attempt, corrections: (
                outputs.append((attempt, corrections)) or ["right"]
            ),
        )

        self.assertEqual(audio, ["right"])
        self.assertTrue(result.passed)
        self.assertEqual(corrections_seen, [(2, [(0, "right", "wrong")])])
        self.assertEqual(outputs, [(2, {"reading": "right"})])


if __name__ == "__main__":
    unittest.main()
