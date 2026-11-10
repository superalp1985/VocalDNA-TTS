from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

import ijson


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "src/vocaldna/data/manifest.json").read_text(encoding="utf-8"))
FORBIDDEN_SUFFIXES = (".pth", ".pt", ".onnx", ".wav", ".safetensors")


def count_lexicon(stream):
    return sum(1 for _ in ijson.kvitems(stream, ""))


def verify_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata_lines = archive.read(metadata_name).splitlines()
        fields = {
            key.decode("ascii"): value.decode("utf-8")
            for line in metadata_lines
            if b": " in line
            for key, value in [line.split(b": ", 1)]
            if key in {b"Author", b"License-Expression"}
        }
        assert fields["Author"] == "王大粘"
        assert fields["License-Expression"] == "LicenseRef-VocalDNA-TTS-Noncommercial-1.0"
        lexicon_name = "vocaldna/data/master_dict.json.gz"
        assert lexicon_name in names
        for name in names:
            assert not name.lower().endswith(FORBIDDEN_SUFFIXES), name
        with archive.open(lexicon_name) as compressed, gzip.GzipFile(fileobj=compressed) as stream:
            assert count_lexicon(stream) == MANIFEST["entries"]
        assert "vocaldna/data/manifest.json" in names


def verify_sdist(path):
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        lexicon_name = next(
            name for name in names if name.endswith("/src/vocaldna/data/master_dict.json.gz")
        )
        compressed = archive.extractfile(lexicon_name)
        with gzip.GzipFile(fileobj=compressed) as stream:
            assert count_lexicon(stream) == MANIFEST["entries"]
        assert any(name.endswith("/LICENSE") for name in names)
        assert any(name.endswith("/NOTICE") for name in names)
        assert any(name.endswith("/tests/test_g2p.py") for name in names)
        assert not any(name.lower().endswith(FORBIDDEN_SUFFIXES) for name in names)


def main():
    wheel = next((ROOT / "dist").glob("*.whl"))
    sdist = next((ROOT / "dist").glob("*.tar.gz"))
    verify_wheel(wheel)
    verify_sdist(sdist)
    print(
        f"Verified wheel and sdist: {MANIFEST['entries']} dictionary entries, "
        "noncommercial license, no model/audio weights."
    )


if __name__ == "__main__":
    main()
