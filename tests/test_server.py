import json
import io
import threading
import unittest
import wave
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.parse import unquote
from urllib.request import Request, urlopen

import numpy as np

from vocaldna.engine import AudioResult
from vocaldna.g2p import G2PTable
from vocaldna.pipeline import VocalDNA
from vocaldna.server import make_handler


class ServerTests(unittest.TestCase):
    def start_server(self, engine=None):
        pipeline = VocalDNA(
            g2p=G2PTable({"会计": {"pinyin": "kuài jì"}}),
            engine=engine,
        )
        server = HTTPServer(("127.0.0.1", 0), make_handler(pipeline))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

        self.addCleanup(cleanup)

        def post(route, body):
            return urlopen(Request(
                f"http://127.0.0.1:{server.server_port}{route}",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ), timeout=3)

        return post

    def test_local_g2p_endpoint_works_without_any_engine(self):
        post = self.start_server()
        with post("/v1/g2p", {"text": "会计"}) as response:
            result = json.loads(response.read())
        self.assertEqual(result["pronunciation"], ["kuai4", "ji4"])

    def test_tts_without_an_engine_returns_service_unavailable(self):
        post = self.start_server()
        with self.assertRaises(HTTPError) as error:
            post("/v1/tts", {"text": "会计"})
        with error.exception as response:
            self.assertEqual(response.code, 503)

    def test_tts_uses_custom_default_and_accepts_engine_specific_voice_ids(self):
        class Engine:
            default_voice = "本地旁白"

            def synthesize(self, text, *, voice, **kwargs):
                return AudioResult(np.zeros(10, np.float32), 16000, voice)

        post = self.start_server(Engine())
        for requested, expected in [(None, "本地旁白"), ("zh-CN:旁白", "zh-CN:旁白")]:
            with self.subTest(requested=requested):
                body = {"text": "会计"}
                if requested is not None:
                    body["voice"] = requested
                with post("/v1/tts", body) as response:
                    self.assertEqual(unquote(response.headers["X-VocalDNA-Voice"]), expected)
                    with wave.open(io.BytesIO(response.read()), "rb") as wav:
                        self.assertEqual(wav.getframerate(), 16000)


if __name__ == "__main__":
    unittest.main()
