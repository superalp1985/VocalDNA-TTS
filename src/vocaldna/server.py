from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

import numpy as np

from .emotion import FAME


MAX_REQUEST_BYTES = 1_048_576


def make_handler(pipeline):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VocalDNA/0.1"

        def _send(self, status, body, content_type="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ValueError("invalid Content-Length") from None
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("request body must be between 1 byte and 1 MiB")
            try:
                data = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ValueError("request body must be valid JSON") from None
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def do_GET(self):
            if self.path == "/health":
                engine = type(pipeline.engine).__name__ if pipeline.engine is not None else None
                return self._send(200, {"ok": True, "engine": engine})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in {"/v1/g2p", "/v1/tts"}:
                return self._send(404, {"error": "not found"})
            try:
                data = self._read_json()
                text = str(data.get("text") or "")
                if not text:
                    raise ValueError("text is required")
                domain = data.get("domain")
                if domain is not None and not (
                    isinstance(domain, str)
                    or (isinstance(domain, list) and all(isinstance(item, str) for item in domain))
                ):
                    raise ValueError("domain must be a string or a list of strings")
                if self.path == "/v1/g2p":
                    tokens = pipeline.g2p.convert(text, domain=domain)
                    return self._send(200, {
                        "text": text,
                        "pronunciation": pipeline.g2p.pronounce(text, domain=domain),
                        "tokens": [
                            {
                                "surface": token.surface,
                                "pinyin": token.pinyin,
                                "kind": token.kind,
                                "metadata": token.metadata,
                            }
                            for token in tokens
                        ],
                    })
                if pipeline.engine is None:
                    return self._send(503, {"error": "no synthesis engine is configured"})
                fame = FAME.from_mapping(data.get("fame"))
                try:
                    speed = float(data.get("speed", 1.0))
                except (TypeError, ValueError):
                    raise ValueError("speed must be a number") from None
                voice = data.get("voice")
                if voice is not None and (
                    not isinstance(voice, str) or not voice.strip() or not voice.isprintable()
                ):
                    raise ValueError("voice must be a nonempty printable string")
                result = pipeline.synthesize(
                    text,
                    voice=voice,
                    speed=speed,
                    fame=fame,
                    domain=domain,
                )
                import soundfile as sf
                output = io.BytesIO()
                sf.write(output, np.asarray(result.audio, dtype=np.float32), result.sample_rate, format="WAV")
                voice_header = quote(result.voice, safe="")
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(output.tell()))
                self.send_header("X-Sample-Rate", str(result.sample_rate))
                self.send_header("X-VocalDNA-Voice", voice_header)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(output.getvalue())
            except ValueError as error:
                self._send(400, {"error": str(error)})
            except Exception:
                self._send(500, {"error": "synthesis failed"})

        def log_message(self, format, *args):
            return

    return Handler


def serve(pipeline, *, host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), make_handler(pipeline))
    try:
        server.serve_forever()
    finally:
        server.server_close()
