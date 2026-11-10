"""VocalDNA: a replaceable, table-first Chinese TTS pipeline."""

from .emotion import FAME, Prosody
from .engine import AudioResult, KokoroEngine, MemoryEngine, SynthesisEngine
from .bert import BertVITS2Engine
from .g2p import G2PTable, LexiconEntry, PronunciationToken
from .pipeline import SynthesisOutput, VocalDNA

__author__ = "王大粘"
__version__ = "0.1.0"

__all__ = [
    "AudioResult",
    "BertVITS2Engine",
    "FAME",
    "G2PTable",
    "KokoroEngine",
    "LexiconEntry",
    "MemoryEngine",
    "PronunciationToken",
    "Prosody",
    "SynthesisEngine",
    "SynthesisOutput",
    "VocalDNA",
]
