from __future__ import annotations

import gzip
import importlib.resources
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


_PUNCTUATION = set("，。！？；：、,.!?;:!?…—-()（）[]【】「」『』\"“”‘’")
_TONE_MARKS = {
    "ā": ("a", 1), "á": ("a", 2), "ǎ": ("a", 3), "à": ("a", 4),
    "ē": ("e", 1), "é": ("e", 2), "ě": ("e", 3), "è": ("e", 4),
    "ī": ("i", 1), "í": ("i", 2), "ǐ": ("i", 3), "ì": ("i", 4),
    "ō": ("o", 1), "ó": ("o", 2), "ǒ": ("o", 3), "ò": ("o", 4),
    "ū": ("u", 1), "ú": ("u", 2), "ǔ": ("u", 3), "ù": ("u", 4),
    "ǖ": ("ü", 1), "ǘ": ("ü", 2), "ǚ": ("ü", 3), "ǜ": ("ü", 4),
    "Ā": ("A", 1), "Á": ("A", 2), "Ǎ": ("A", 3), "À": ("A", 4),
    "Ē": ("E", 1), "É": ("E", 2), "Ě": ("E", 3), "È": ("E", 4),
    "Ō": ("O", 1), "Ó": ("O", 2), "Ǒ": ("O", 3), "Ò": ("O", 4),
}
_PINYIN_RE = re.compile(r"[A-Za-züÜvV:]+(?:[1-5])?$")


@dataclass(frozen=True)
class LexiconEntry:
    surface: str
    pinyin: list[str]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PronunciationToken:
    surface: str
    pinyin: list[str]
    kind: str = "word"
    metadata: dict[str, object] = field(default_factory=dict)


def syllable_to_number(value: str) -> str:
    """Convert tone marks or an existing tone number to a stable TONE3 form."""
    value = str(value).strip()
    if not value:
        return value
    from pypinyin.contrib.tone_convert import to_finals, to_tone3

    if value[-1:] in "12345":
        return value.replace("ü", "v")
    tone = 5
    base = []
    for char in value:
        if char in _TONE_MARKS:
            plain, tone = _TONE_MARKS[char]
            base.append(plain)
        else:
            base.append(char)
    plain = "".join(base)
    if value == value.lower() and to_finals(value):
        return to_tone3(value, neutral_tone_with_five=True).replace("ü", "v")
    return plain + (str(tone) if tone != 5 else "")


def _is_pinyin(value: str) -> bool:
    return bool(_PINYIN_RE.fullmatch(value))


def _change_tone(syllable: str, tone: int) -> str:
    if not syllable or syllable[-1:] not in "12345":
        return syllable
    return syllable[:-1] + str(tone)


def apply_tone_sandhi(syllables: Iterable[str], characters: Iterable[str] | None = None) -> list[str]:
    result = list(syllables)
    original = list(result)
    chars = list(characters) if characters is not None else [""] * len(result)
    for index, syllable in enumerate(result):
        if chars[index] == "不" and index + 1 < len(result):
            next_syllable = result[index + 1]
            if next_syllable[-1:] == "4":
                result[index] = _change_tone(syllable, 2)
            else:
                result[index] = _change_tone(syllable, 4)
        if chars[index] == "一" and index + 1 < len(result):
            if index and chars[index - 1] == "第":
                continue
            next_tone = result[index + 1][-1:]
            result[index] = _change_tone(syllable, 2 if next_tone == "4" else 4 if next_tone in "123" else 1)
    for index in range(len(result) - 1):
        if original[index].endswith("3") and original[index + 1].endswith("3"):
            result[index] = _change_tone(result[index], 2)
    return result


class G2PTable:
    """Longest-match lexicon lookup with deterministic fallback and tone sandhi."""

    def __init__(
        self,
        entries: Mapping[str, Mapping[str, object] | LexiconEntry] | Iterable[tuple[str, Mapping[str, object] | LexiconEntry]] | None = None,
        *,
        overrides: Mapping[str, str | Iterable[str]] | None = None,
        fallback: bool = True,
    ):
        self._entries: dict[str, LexiconEntry] = {}
        self._overrides: dict[str, LexiconEntry] = {}
        self.fallback = fallback
        source_entries = entries.items() if isinstance(entries, Mapping) else (entries or ())
        for surface, entry in source_entries:
            self._entries[str(surface)] = self._coerce_entry(str(surface), entry)
        for surface, pinyin in (overrides or {}).items():
            if isinstance(pinyin, str):
                values = pinyin.split()
            else:
                values = list(pinyin)
            self._overrides[str(surface)] = LexiconEntry(
                surface=str(surface),
                pinyin=[syllable_to_number(item) for item in values],
                metadata={"override": True, "source": "override"},
            )
        self._lengths = sorted(
            {len(key) for key in (*self._entries.keys(), *self._overrides.keys()) if key},
            reverse=True,
        )

    @staticmethod
    def _coerce_entry(surface: str, entry: Mapping[str, object] | LexiconEntry) -> LexiconEntry:
        if isinstance(entry, LexiconEntry):
            return entry
        raw = entry.get("pinyin", "")
        raw = raw.split() if isinstance(raw, str) else list(raw)
        retained = {
            key: value
            for key, value in entry.items()
            if key in {"domain", "pos", "pos_code", "category", "emotion", "scene", "is_polyphone"}
        }
        return LexiconEntry(
            surface=surface,
            pinyin=[syllable_to_number(item) for item in raw],
            metadata=retained,
        )

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> "G2PTable":
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            import ijson
        except ImportError:
            with opener(path, "rt", encoding="utf-8") as stream:
                data = json.load(stream)
            if not isinstance(data, dict):
                raise ValueError("lexicon JSON must contain an object")
            return cls(data, **kwargs)
        with opener(path, "rb") as stream:
            return cls(ijson.kvitems(stream, ""), **kwargs)

    @classmethod
    def from_resource(cls, path: str | Path | None = None, **kwargs) -> "G2PTable":
        if path is not None:
            return cls.from_json(Path(path), **kwargs)
        resource = importlib.resources.files("vocaldna").joinpath("data/master_dict.json.gz")
        with importlib.resources.as_file(resource) as resource_path:
            return cls.from_json(resource_path, **kwargs)

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, surface: str, *, domain: str | Iterable[str] | None = None) -> LexiconEntry | None:
        if surface in self._overrides:
            return self._overrides[surface]
        entry = self._entries.get(surface)
        if entry is None:
            return None
        if domain is None:
            return entry
        allowed = {domain} if isinstance(domain, str) else set(domain)
        entry_domain = entry.metadata.get("domain")
        return entry if entry_domain is None or entry_domain in allowed or entry_domain == "通用" else None

    def _fallback_entry(self, surface: str) -> LexiconEntry | None:
        if not self.fallback:
            return None
        try:
            from pypinyin import Style, lazy_pinyin
        except ImportError:
            return None
        values = lazy_pinyin(surface, style=Style.TONE3, neutral_tone_with_five=True)
        if not values:
            return None
        return LexiconEntry(surface, [syllable_to_number(item) for item in values], {"source": "pypinyin"})

    def convert(self, text: str, *, domain: str | Iterable[str] | None = None) -> list[PronunciationToken]:
        text = str(text or "")
        tokens: list[PronunciationToken] = []
        index = 0
        while index < len(text):
            char = text[index]
            if char.isspace():
                index += 1
                continue
            if char in _PUNCTUATION:
                tokens.append(PronunciationToken(char, [], "punctuation", {"boundary": True}))
                index += 1
                continue
            match: LexiconEntry | None = None
            for length in self._lengths:
                if index + length > len(text):
                    continue
                candidate = text[index:index + length]
                match = self.lookup(candidate, domain=domain)
                if match is not None:
                    break
            if match is None:
                candidate = char
                match = self.lookup(candidate, domain=domain) or self._fallback_entry(candidate)
                if match is None:
                    tokens.append(PronunciationToken(candidate, [], "text", {"source": "unmapped"}))
                else:
                    tokens.append(PronunciationToken(candidate, match.pinyin, "fallback", match.metadata))
                index += 1
                continue
            metadata = dict(match.metadata)
            metadata.setdefault("matched", True)
            tokens.append(PronunciationToken(match.surface, list(match.pinyin), "word", metadata))
            index += len(match.surface)
        return tokens

    def apply_sandhi(self, tokens: list[PronunciationToken]) -> list[PronunciationToken]:
        result = []
        phrase = []

        def flush():
            syllables = [p for tk in phrase for p in tk.pinyin]
            characters = [
                char
                for tk in phrase
                for char in (list(tk.surface) if len(tk.surface) == len(tk.pinyin) else [""] * len(tk.pinyin))
            ]
            adjusted = apply_tone_sandhi(syllables, characters)
            start = 0
            for tk in phrase:
                end = start + len(tk.pinyin)
                result.append(PronunciationToken(tk.surface, adjusted[start:end], tk.kind, tk.metadata))
                start = end
            phrase.clear()

        for token in tokens:
            if token.kind == "punctuation":
                flush()
                result.append(token)
                continue
            phrase.append(token)
        flush()
        return result

    def pronounce(self, text: str, *, domain: str | Iterable[str] | None = None) -> list[str]:
        return [p for tk in self.apply_sandhi(self.convert(text, domain=domain)) for p in tk.pinyin]
