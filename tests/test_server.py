import json
import threading
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen

import numpy as np

from vocaldna.engine import AudioResult
from vocaldna.g2p import G2PTable
from vocaldna.pipeline import VocalDNA
from vocaldna.server import make_handler


class ServerTests(unittest.TestCase):
    def test_local_g2p_endpoint_returns_table_pronunciation(self):
        class Engine:
            sample_rate = 24000

            def synthesize(self, text, **kwargs):
                return AudioResult(np.zeros(10, np.float32), 24000, kwargs["voice"])

        pipeline = VocalDNA(
            g2p=G2PTable({"会计": {"pinyin": "kuài jì"}}),
            engine=Engine(),
        )
        server = HTTPServer(("127.0.0.1", 0), make_handler(pipeline))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{server.server_port}/v1/g2p",
                data=json.dumps({"text": "会计"}, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=3) as response:
                result = json.loads(response.read())
            self.assertEqual(result["pronunciation"], ["kuai4", "ji4"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
