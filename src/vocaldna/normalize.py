from __future__ import annotations

import re


_LETTER_READ = {
    "A": "诶", "B": "比", "C": "西", "D": "迪", "E": "伊", "F": "艾弗",
    "G": "吉", "H": "艾奇", "I": "爱", "J": "杰", "K": "开", "L": "艾尔",
    "M": "艾姆", "N": "恩", "O": "欧", "P": "批", "Q": "克尤", "R": "阿尔",
    "S": "艾斯", "T": "踢", "U": "优", "V": "维", "W": "达布尤", "X": "艾克斯",
    "Y": "歪", "Z": "孜",
}
_ABBREVIATIONS = {
    "AI": "诶爱", "API": "诶批爱", "APP": "诶批批", "ASR": "诶艾斯阿尔",
    "TTS": "踢踢艾斯", "CPU": "西批优", "GPU": "吉批优", "NPU": "恩批优",
    "USB": "优艾斯比", "SSD": "艾斯艾斯迪", "HTML": "艾奇踢艾姆艾尔",
    "PDF": "批迪艾弗", "ONNX": "欧恩恩艾克斯", "CUDA": "库达", "KOKORO": "扣扣若",
}
_UNIT_READ = {
    "kg": "千克", "g": "克", "mg": "毫克", "km": "公里", "m": "米",
    "cm": "厘米", "mm": "毫米", "Hz": "赫兹", "kHz": "千赫兹",
    "MHz": "兆赫兹", "GHz": "吉赫兹", "GB": "吉字节", "MB": "兆字节",
    "KB": "千字节", "TB": "太字节", "fps": "帧每秒", "ms": "毫秒", "dB": "分贝",
}
_SYMBOL_READ = {
    "——": "，", "……": "，", "...": "，", "、": "，", "→": "到", "←": "从",
    "+": "加", "×": "乘", "÷": "除以", "=": "等于", "≠": "不等于",
    "≈": "约等于", "≤": "小于等于", "≥": "大于等于", "℃": "摄氏度",
    "°C": "摄氏度", "°F": "华氏度", "°": "度", "¥": "元", "$": "美元",
    "€": "欧元", "£": "英镑", "%": "百分之", "&": "和", "@": "艾特", "#": "井号",
}
_DIGITS = "零一二三四五六七八九"


class TextNormalizer:
    """Normalize numbers, units and technical abbreviations before synthesis."""

    def __init__(self, *, extra_terms: dict[str, str] | None = None):
        self.terms = dict(_ABBREVIATIONS)
        self.terms.update({str(key).upper(): str(value) for key, value in (extra_terms or {}).items()})

    def _num_to_cn(self, value: str) -> str:
        import cn2an

        value = str(value).lstrip("0") or "0"
        if len(value) > 16:
            return "".join(_DIGITS[int(char)] for char in value)
        return cn2an.an2cn(value, "low")

    def _format_number(self, value: str) -> str:
        if "." not in value:
            return self._num_to_cn(value)
        whole, fraction = value.split(".", 1)
        return self._num_to_cn(whole) + "点" + "".join(_DIGITS[int(char)] for char in fraction)

    def _spell(self, token: str) -> str:
        upper = token.upper()
        if upper in self.terms:
            return self.terms[upper]
        result = []
        for char in upper:
            if char.isalpha():
                result.append(_LETTER_READ.get(char, char))
            elif char.isdigit():
                result.append(_DIGITS[int(char)])
            elif char in "+-_.:":
                result.append({"+": "加", "-": "杠", "_": "杠", ".": "点", ":": "冒"}[char])
        return "".join(result) or token

    def _replace_technical_tokens(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            version = re.fullmatch(r"([a-z])(\d+(?:\.\d+)?)", token)
            if version:
                return self._spell(version.group(1)) + self._format_number(version.group(2))
            return self._spell(token)

        return re.sub(r"[A-Za-z][A-Za-z0-9+_.:-]*", replace, text)

    def normalize(self, text: str) -> str:
        text = str(text or "")
        text = re.sub(
            r"(\d+(?:\.\d+)?)\s*%",
            lambda match: "百分之" + self._format_number(match.group(1)),
            text,
        )
        text = re.sub(
            r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",
            lambda match: self._format_number(match.group(2)) + "分之" + self._format_number(match.group(1)),
            text,
        )
        for unit, reading in sorted(_UNIT_READ.items(), key=lambda item: -len(item[0])):
            text = re.sub(
                r"(\d+(?:\.\d+)?)\s*" + re.escape(unit) + r"(?![A-Za-z])",
                lambda match, reading=reading: self._format_number(match.group(1)) + reading,
                text,
            )
        text = self._replace_technical_tokens(text)
        text = re.sub(
            r"(\d+)\.(\d+)",
            lambda match: self._num_to_cn(match.group(1)) + "点" + "".join(_DIGITS[int(char)] for char in match.group(2)),
            text,
        )
        text = re.sub(r"\d+", lambda match: self._num_to_cn(match.group(0)), text)
        for symbol, reading in sorted(_SYMBOL_READ.items(), key=lambda item: -len(item[0])):
            text = text.replace(symbol, reading)
        return re.sub(r"\s+", "", text).strip()
