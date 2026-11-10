from __future__ import annotations

from typing import Iterable

from .g2p import PronunciationToken


def split_syllable(syllable: str) -> tuple[str, str, int]:
    from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

    value = str(syllable).lower()
    if not value:
        raise ValueError("empty pinyin syllable")
    initial = to_initials(value)
    final_with_tone = to_finals_tone3(value, neutral_tone_with_five=True)
    tone = int(final_with_tone[-1])
    final = final_with_tone[:-1]
    if final == "i" and initial in {"z", "c", "s"}:
        final = "ii"
    elif final == "i" and initial in {"zh", "ch", "sh", "r"}:
        final = "iii"
    return initial, final, tone


def kokoro_phone_units(tokens: Iterable[PronunciationToken]) -> list[str]:
    from misaki.zh_frontend import ZH_MAP

    units = []
    for token in tokens:
        if token.kind == "punctuation":
            units.append({
                "。": ".", "，": ",", "、": ",", "！": "!", "？": "?",
                "；": ";", "：": ":", "…": "…", "—": "—",
            }.get(token.surface, token.surface))
            continue
        for syllable in token.pinyin:
            initial, final, tone = split_syllable(syllable)
            try:
                units.append(ZH_MAP.get(initial, "") + ZH_MAP[final] + str(tone))
            except KeyError:
                raise ValueError(f"unsupported Mandarin syllable: {syllable}") from None
    return units
