"""Спецификация российских ГРЗ: алфавит, маски номеров, коды регионов, геометрия по ГОСТ.

Единственный источник истины для всего проекта: генератора, модели и валидатора.
Соответствует evaluate.py / validate_dataset.py организаторов (Волга-IT 2026, полуфинал).
"""

from __future__ import annotations

import random
import re


# Буквы, совпадающие по начертанию в кириллице и латинице (ГОСТ Р 50577-2018).
LATIN_LETTERS = "ABEKMHOPCTYX"
CYR_LETTERS = "АВЕКМНОРСТУХ"
CYR_TO_LAT = str.maketrans(CYR_LETTERS + CYR_LETTERS.lower(), LATIN_LETTERS + LATIN_LETTERS)

PLATE_TYPES = ("type1", "type1a", "type1b", "other")
TARGET_TYPES = ("type1", "type1a", "type1b")

# Классы YOLO для labels/*.txt (порядок задан валидатором).
LABEL_CLASSES = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}
CLASS_TO_LABEL = {v: k for k, v in LABEL_CLASSES.items()}

# Допустимые поля в meta.csv (порядок фиксирован валидатором).
CONDITIONS = ("day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle")

META_COLUMNS = (
    "image", "plate_num", "plate_type", "bbox", "quad",
    "is_vehicle", "is_synthetic", "source", "license", "conditions",
)


def _mask(body: str) -> re.Pattern[str]:
    L = f"[{LATIN_LETTERS}#]"
    D = "[0-9#]"
    return re.compile(f"^{body}$")


#   type1, type1a : Л ЦЦЦ ЛЛ РР(Р)   → A123BC777
#   type1b        : ЛЛ ЦЦЦ РР(Р)     → AB12377
#   other         : произвольная строка из A-Z, 0-9, «#»
PLATE_MASKS = {
    "type1": _mask(f"[{LATIN_LETTERS}#][0-9#]{{3}}[{LATIN_LETTERS}#]{{2}}[0-9#]{{2,3}}"),
    "type1a": _mask(f"[{LATIN_LETTERS}#][0-9#]{{3}}[{LATIN_LETTERS}#]{{2}}[0-9#]{{2,3}}"),
    "type1b": _mask(f"[{LATIN_LETTERS}#]{{2}}[0-9#]{{3}}[0-9#]{{2,3}}"),
    "other": re.compile(r"^[A-Z0-9#]{0,12}$"),
}

# Длина «тела» номера без кода региона — нужно для отделения региона и для CTC.
BODY_LEN = {"type1": 6, "type1a": 6, "type1b": 5, "other": 6}

# Геометрия по ГОСТ Р 50577-2018, мм (ширина × высота).
PLATE_MM = {"type1": (520, 112), "type1a": (290, 170), "type1b": (520, 112)}


class PlateType:
    """Описание типа знака: раскладка по строкам, пропорции, цвет фона."""

    __slots__ = ("name", "rows", "aspect", "bg", "fg")

    def __init__(self, name: str, rows: int, aspect: float, bg: str):
        self.name = name
        self.rows = rows
        self.aspect = aspect
        self.bg = bg
        self.fg = "black"

    @property
    def mask(self) -> re.Pattern[str]:
        return PLATE_MASKS[self.name]

    @property
    def body_len(self) -> int:
        return BODY_LEN[self.name]

    def split_visual(self, plate: str) -> list[str]:
        """Разбивает канонический номер на строки в порядке чтения сверху вниз.

        Ключевое свойство формата: при чтении «слева направо, сверху вниз» визуальный
        порядок символов СОВПАДАЕТ с каноническим для всех трёх целевых типов.
            type1  → одна строка: A123BC777
            type1a → две строки: 'A123' + 'BC777'
            type1b → одна строка: AB12377
        """
        b = self.body_len
        if self.rows == 1:
            return [plate]
        # Двухстрочный (type1a): верх — буква + 3 цифры, низ — 2 буквы + регион.
        top = 4
        return [plate[:top], plate[top:]]


TYPES = {
    "type1": PlateType("type1", rows=1, aspect=520 / 112, bg="white"),
    "type1a": PlateType("type1a", rows=2, aspect=290 / 170, bg="white"),
    "type1b": PlateType("type1b", rows=1, aspect=520 / 112, bg="yellow"),
}

# ---------------------------------------------------------------------------
# Коды регионов
# ---------------------------------------------------------------------------

REGION_CODES_2 = (
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14",
    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28",
    "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "43",
    "44", "45", "46", "47", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58",
    "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72",
    "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86",
    "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98",
)

# Трёхзначные коды с новыми регионами и новыми типами знаков.
REGION_CODES_3 = (
    "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "120",
    "121", "122", "123", "124", "125", "126", "127", "128", "129", "130",
    "131", "132", "133", "134", "135", "136", "137", "138", "139",
    "201", "202", "203", "204", "205", "206", "207", "208", "209",
    "251", "252", "253", "254", "255", "256", "257", "258", "259",
    "301", "302", "303", "304", "305", "306", "307", "308", "309",
    "401", "402",
    "701", "702", "703", "704", "705", "706", "707", "708", "709",
    "750", "751", "752", "753", "754", "755", "756", "757", "758",
    "901", "902", "903", "904", "905", "906", "907", "908", "909",
)

ALL_REGION_CODES = frozenset(REGION_CODES_2) | frozenset(REGION_CODES_3)

# Регионы, где живёт основная масса машин — для реалистичного распределения.
_WEIGHTED_REGIONS = (
    [c for c in REGION_CODES_2 if c in {"77", "50", "16", "23", "01", "02", "03", "04",
                                        "05", "06", "07", "14", "16", "17", "18", "19",
                                        "20", "21", "25", "26", "27", "29", "30", "31",
                                        "32", "34", "37", "40", "41", "42", "43", "44",
                                        "45", "46", "47", "48", "49", "50", "52", "53",
                                        "54", "55", "56", "57", "58", "59", "60", "61",
                                        "62", "63", "64", "65", "66", "67", "68", "69",
                                        "70", "71", "72", "73", "74", "75", "76"}]
    + list(REGION_CODES_3)
)


# ---------------------------------------------------------------------------
# Генерация и проверка номеров
# ---------------------------------------------------------------------------

def random_region(rng: random.Random, weighted: bool = True) -> str:
    pool = _WEIGHTED_REGIONS if weighted else sorted(ALL_REGION_CODES)
    return rng.choice(pool)


def random_plate(rng: random.Random, ptype: str) -> str:
    """Случайный корректный номер заданного типа."""
    b = TYPES[ptype].body_len
    body: list[str] = []
    if ptype == "type1b":
        # ЛЛ ЦЦЦ
        body = [rng.choice(LATIN_LETTERS), rng.choice(LATIN_LETTERS)]
        body += [rng.choice("0123456789") for _ in range(3)]
    else:
        # Л ЦЦЦ ЛЛ
        body = [rng.choice(LATIN_LETTERS)]
        body += [rng.choice("0123456789") for _ in range(3)]
        body += [rng.choice(LATIN_LETTERS), rng.choice(LATIN_LETTERS)]
    assert len(body) == b, (ptype, body)
    return "".join(body) + random_region(rng)


def mask_positions(plate: str, ptype: str) -> list[tuple[int, str]]:
    """Позиции, на которых в маске стоит буква: [(index, 'L'|'D'), ...]."""
    b = TYPES[ptype].body_len
    out: list[tuple[int, str]] = []
    i = 0
    if ptype == "type1b":
        out += [(i, "L"), (i + 1, "L")]
        i += 2
        out += [(i, "D"), (i + 1, "D"), (i + 2, "D")]
        i += 3
    else:
        out.append((i, "L"))
        i += 1
        out += [(i, "D"), (i + 1, "D"), (i + 2, "D")]
        i += 3
        out += [(i, "L"), (i + 1, "L")]
        i += 2
    out += [(i + k, "D") for k in range(len(plate) - i)]
    return out


def is_valid_plate(plate: str, ptype: str) -> bool:
    return bool(PLATE_MASKS[ptype].fullmatch(plate))


def body_of(plate: str, ptype: str) -> str:
    return plate[: TYPES[ptype].body_len]


def region_of(plate: str, ptype: str) -> str:
    return plate[TYPES[ptype].body_len:]
