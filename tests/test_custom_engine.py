import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from vocaldna import AudioResult, G2PTable, VocalDNA
from vocaldna.cli import main


class CustomEngine:
    default_voice = "narrator"

    def synthesize(self, text, **kwargs):
        tokens = kwargs["pronunciation_tokens"]
        return AudioResult(
            np.full(100, 0.1, dtype=np.float32),
            16000,
            kwargs["voice"],
            {"firstToken": tokens[0].surface, "receivedFame": kwargs["fame"].to_vector()},
        )


def make_engine():
    return CustomEngine()


class CustomEngineTests(unittest.TestCase):
    def test_pipeline_passes_tokens_and_uses_engine_default_voice(self):
        pipeline = VocalDNA(g2p=G2PTable(), engine=CustomEngine())
        result = pipeline.synthesize("\u4f60", fame={"calm": 0.4})

        self.assertEqual(result.voice, "narrator")
        self.assertEqual(result.metadata["firstToken"], "\u4f60")
        self.assertIn(0.4, result.metadata["receivedFame"])
        self.assertEqual(next(pipeline.synthesize_stream("\u4f60")).result.voice, "narrator")

    def test_cli_loads_custom_engine_and_writes_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            lexicon = Path(tmp) / "lexicon.json"
            lexicon.write_text("{}", encoding="utf-8")
            wav = Path(tmp) / "output.wav"
            args = [
                "vocaldna", "synthesize", "\u4f60", "--engine", f"{__name__}:make_engine",
                "--lexicon", str(lexicon), "--output", str(wav),
            ]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()) as stdout:
                main()

            self.assertEqual(sf.info(wav).samplerate, 16000)
            self.assertEqual(json.loads(stdout.getvalue())["voice"], "narrator")

    def test_serve_uses_the_same_custom_engine_option(self):
        for options, engine in [([], None), (["--engine", f"{__name__}:make_engine"], CustomEngine)]:
            with (
                self.subTest(options=options),
                patch("vocaldna.cli._table", return_value=G2PTable()),
                patch("vocaldna.server.serve") as serve,
                patch.object(sys, "argv", ["vocaldna", "serve", *options]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main()

                selected = serve.call_args.args[0].engine
                if engine is None:
                    self.assertIsNone(selected)
                else:
                    self.assertIsInstance(selected, engine)

    def test_cli_synthesis_requires_an_engine_choice(self):
        with patch.object(sys, "argv", ["vocaldna", "synthesize", "\u4f60"]):
            with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as error:
                main()

        self.assertEqual(error.exception.code, 2)
        self.assertIn("--engine", stderr.getvalue())
