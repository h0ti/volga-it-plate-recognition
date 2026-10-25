"""Рендер номеров по ГОСТ Р 50577-2018 и затем наружный вид кадра.

Модуль не зависит от сети и использует только системные шрифты — генератор
полностью воспроизводим (фиксированный random seed) и не тянет внешние ассеты.
"""

from __future__ import annotations

import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import spec

FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")

# Системные шрифты с кириллицей. Используются только для рендера, в репозиторий
# не копируются (чтобы не нарушать лицензии), — генератор указывает их в README.
FONT_FILES = (
    "arial.ttf", "arialbd.ttf", "arialn.ttf", "tahoma.ttf", "tahomabd.ttf",
    "verdana.ttf", "verdanab.ttf", "calibri.ttf", "calibrib.ttf", "segoeui.ttf",
    "segoeuib.ttf", "impact.ttf", "trebuc.ttf", "cour.ttf", "micross.ttf",
    "framd.ttf", "GARA.TTF", "BRLNSB.TTF",
)

# Фон знака по типу. Жёлтый — не чистый жёлтый, а смотровое покрытие.
BG_WHITE = (248, 248, 246)
BG_YELLOW = (226, 196, 26)
FG_BLACK = (18, 18, 20)


def available_fonts() -> list[str]:
    out = []
    for f in FONT_FILES:
        p = os.path.join(FONT_DIR, f)
        if os.path.isfile(p):
            out.append(p)
    return out


_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


# ---------------------------------------------------------------------------
# Пластина знака
# ---------------------------------------------------------------------------

def render_plate(
    plate: str,
    ptype: str,
    width_px: int,
    rng: random.Random,
    *,
    draw_flag: bool = False,
) -> Image.Image:
    """Рисует пластину знака заданного типа размером width_px × (width_px / aspect).

    Возвращает RGBA-изображение пластины с непрозрачным фоном и рамкой.
    """
    t = spec.TYPES[ptype]
    mm_w, mm_h = spec.PLATE_MM[ptype]
    h_px = max(8, int(round(width_px * mm_h / mm_w)))
    scale = width_px / mm_w

    img = Image.new("RGB", (width_px, h_px), BG_WHITE if t.bg == "white" else BG_YELLOW)
    d = ImageDraw.Draw(img)

    # Небольшой разброс фона — знаки не бывают идеально ровными.
    bg = BG_WHITE if t.bg == "white" else BG_YELLOW
    jitter = rng.randint(-8, 8)
    bg = tuple(max(0, min(255, c + jitter)) for c in bg)
    d.rectangle([0, 0, width_px, h_px], fill=bg)

    fonts = available_fonts()
    if not fonts:
        raise RuntimeError("Не найдены системные шрифты для рендера номеров")

    rows = t.split_visual(plate)
    sym_h = int(round(50 * scale))            # высота символа 50 мм по ГОСТ

    # Отступы: поля знака ~ 10 мм по бокам (для 1А — 15 мм).
    if ptype == "type1a":
        pad_x = int(round(15 * scale))
        pad_y = int(round(12 * scale))
        gap = int(round(10 * scale))
        flag_w = int(round(38 * scale)) if draw_flag else 0
    else:
        pad_x = int(round(12 * scale))
        pad_y = int(round((112 - 50) / 2 * scale))
        gap = int(round(22 * scale))
        flag_w = 0

    inner_w = width_px - 2 * pad_x - flag_w
    inner_h = h_px - 2 * pad_y

    row_h = sym_h
    total_h = len(rows) * row_h + (len(rows) - 1) * gap
    y0 = pad_y + (inner_h - total_h) // 2

    for ri, row in enumerate(rows):
        # Регион отделяем от тела чуть большим интервалом.
        body = row[:t.body_len] if len(rows) == 1 else (
            row[:4] if ri == 0 else row[:2]
        )
        region = row[len(body):]
        if not region:
            body, region = row, ""

        font = get_font(rng.choice(fonts), sym_h)
        widths = []
        for ch in body:
            bb = d.textbbox((0, 0), ch, font=font)
            widths.append(bb[2] - bb[0])
        reg_font = font
        reg_w = 0
        if region:
            reg_w = sum(
                d.textbbox((0, 0), ch, font=reg_font)[2]
                - d.textbbox((0, 0), ch, font=reg_font)[0]
                for ch in region
            )

        sp = int(round(6 * scale))                       # межсимвольный интервал
        g = gap
        total_w = sum(widths) + sp * (len(body) - 1) + (g + reg_w if region else 0)
        # Обе строки типа 1А выравниваются по левому краю; флаг занимает правый
        # верхний угол и сдвигать верхнюю строку не должен (он и так короче).
        x = pad_x if len(rows) == 2 else pad_x + (inner_w - total_w) // 2

        y = y0 + ri * (row_h + gap)
        bb0 = d.textbbox((0, 0), body[0], font=font)
        # Вертикально центрируем символ в строке по её высоте.
        dy = y + (row_h - (bb0[3] - bb0[1])) // 2 - bb0[1]

        for ch, w in zip(body, widths):
            d.text((x, dy), ch, font=font, fill=FG_BLACK)
            x += w + sp
        if region:
            x += g
            for ch in region:
                bb = d.textbbox((0, 0), ch, font=reg_font)
                d.text((x, dy), ch, font=reg_font, fill=FG_BLACK)
                x += bb[2] - bb[0] + sp

    # Государственный флаг «RUS» (не входит в plate_num).
    if draw_flag:
        try:
            fs = max(6, int(round(11 * scale)))
            fpath = rng.choice(fonts)
            fnt = get_font(fpath, fs)
            tw = d.textbbox((0, 0), "RUS", font=fnt)
            d.rectangle(
                [width_px - flag_w + int(3 * scale), int(4 * scale),
                 width_px - int(4 * scale), int(4 * scale) + (tw[3] - tw[1]) + 4],
                fill=(255, 255, 255),
            )
            d.text((width_px - flag_w + int(5 * scale), int(4 * scale)),
                   "RUS", font=fnt, fill=FG_BLACK)
        except Exception:
            pass

    # Рамка/кромка: у реальных знаков виден тёмный или хромированный кант.
    bw = max(1, int(round(2 * scale)))
    edge = rng.randint(90, 170)
    d.rectangle([0, 0, width_px - 1, h_px - 1], outline=(edge, edge, edge), width=bw)
    return img


# ---------------------------------------------------------------------------
# Перспективные искажения
# ---------------------------------------------------------------------------

def random_perspective(img: Image.Image, rng: random.Random,
                       strength: float = 1.0) -> tuple[Image.Image, list[tuple[float, float]]]:
    """Случайная гомография методом четырёх углов.

    Изображение предварино расширяется полями: без этого искажение вытягивает
    пиксели за пределы исходника, и по краям знака СРЕЗАЮТСЯ символы. При этом
    метка остаётся полной, и модель учится на несогласованных данных — это
    выглядит как «модель не учится», хотя данные сломаны.

    Возвращает (картинка, quad преобразованных углов в координатах результата).
    """
    w, h = img.size
    mx, my = w * 0.16 * strength, h * 0.20 * strength
    pad = int(math.ceil(max(mx, my))) + 2
    # Заливка полей — цветом ФОНА пластины (берём сверху по центру), иначе
    # вокруг знака появилась бы тёмная кайма и модель её выучила бы.
    bg = img.getpixel((w // 2, min(3, h - 1)))
    canvas = Image.new(img.mode, (w + 2 * pad, h + 2 * pad), bg)
    canvas.paste(img, (pad, pad))
    cw, ch = canvas.size
    # Углы по часовой стрелке от левого верхнего, со случайным сдвигом.
    j = lambda m: rng.uniform(-m, m)
    dst = [
        (pad + j(mx), pad + j(my)),
        (pad + cw - pad + j(mx), pad + j(my)),
        (pad + cw - pad + j(mx), pad + ch - pad + j(my)),
        (pad + j(mx), pad + ch - pad + j(my)),
    ]
    src = [(0.0, 0.0), (float(cw), 0.0), (float(cw), float(ch)), (0.0, float(ch))]
    coeffs = _perspective_coeffs(dst, src)
    out = canvas.transform((cw, ch), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    # Кадрируем обратно; углы должны быть в координатах уже обрезанного кадра.
    quad = [(x - pad, y - pad) for x, y in dst]
    return out.crop((pad, pad, pad + w, pad + h)), quad


def _perspective_coeffs(pa: list[tuple[float, float]], pb: list[tuple[float, float]]):
    """Коэффициенты PIL PERSPECTIVE, переводящие pa -> pb."""
    import numpy as np
    a = []
    b = []
    for (x1, y1), (x2, y2) in zip(pa, pb):
        a.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1])
        b.append(x2)
        a.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1])
        b.append(y2)
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    res = np.linalg.solve(A, B)
    return tuple(res)


def apply_quad(img: Image.Image, quad: list[tuple[float, float]],
               out_w: int) -> Image.Image:
    """Perspective-выпрямление кропа по четырём точкам в размер out_w × out_h."""
    import numpy as np
    w, h = img.size
    out_h = max(8, int(round(out_w * h / max(w, 1))))
    src = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    coeffs = _perspective_coeffs([tuple(map(float, p)) for p in quad], src)
    return img.transform((out_w, out_h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


# ---------------------------------------------------------------------------
# Фотографические условия (соответствуютConditions в meta.csv)
# ---------------------------------------------------------------------------

def add_glare(img: Image.Image, rng: random.Random) -> Image.Image:
    import numpy as np
    w, h = img.size
    a = np.asarray(img).astype(np.float32)
    cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = rng.uniform(0.20, 0.65) * max(w, h)
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    a += g[..., None] * rng.uniform(35, 120)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def add_dirt(img: Image.Image, rng: random.Random) -> Image.Image:
    """Локальные загрязнения: пятна грязи/соли точечно, а не по всему знаку.

    Сила подобрана так, чтобы знак оставался читаемым — в задании разметку ставит
    человек, который читает в том числе грязные и бликующие знаки.
    """
    import numpy as np
    w, h = img.size
    a = np.asarray(img).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    layer = np.zeros((h, w), np.float32)
    n = rng.randint(6, 26)
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.01, 0.07) * max(w, h)
        layer = np.maximum(layer, np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * r * r))))
    k = rng.uniform(0.10, 0.32)
    a *= (1 - k * layer[..., None] * 0.55)
    a += layer[..., None] * rng.uniform(-6, 14)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def add_motion_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    """Направленное размытие (смаз от движения) — ядро свёртки вдоль случайной оси."""
    import numpy as np
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    k = rng.choice([5, 7, 9, 11, 13])
    ang = rng.uniform(0, math.pi)
    dx, dy = math.cos(ang), math.sin(ang)
    acc = np.zeros_like(a)
    wsum = 0.0
    for i in range(-(k // 2), k // 2 + 1):
        wt = 1.0 - abs(i) / (k / 2 + 1)
        sx, sy = int(round(i * dx)), int(round(i * dy))
        acc += wt * np.roll(np.roll(a, sy, axis=0), sx, axis=1)
        wsum += wt
    a = acc / max(wsum, 1e-6)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def make_night(img: Image.Image, rng: random.Random) -> Image.Image:
    import numpy as np
    a = np.asarray(img).astype(np.float32)
    g = rng.uniform(0.25, 0.55)
    a = a * g
    # Фонари и блики от фар.
    for _ in range(rng.randint(1, 4)):
        h, w = a.shape[:2]
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(0.15, 0.5) * max(w, h)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        k = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
        a += k[..., None] * rng.uniform(40, 130)
    a += np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, 5, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def add_rain_snow(img: Image.Image, rng: random.Random, snow: bool = False) -> Image.Image:
    """Осадки: тонкие штрихи дождя или мелкие хлопья снега (1–3 px).

    Размер частиц намеренно мал: крупные «блоки» выглядят как артефакт, а не
    как погода, и ломают обучение детектора.
    """
    import numpy as np
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    rs = np.random.default_rng(rng.randint(0, 2**31 - 1))
    if snow:
        n = int(h * w * 0.00035)
        a += (rs.random((h, w, 1)) - 0.5) * 30 + 8
        for _ in range(max(1, n)):
            x, y = int(rs.uniform(0, w - 2)), int(rs.uniform(0, h - 2))
            v = float(rs.uniform(150, 250))
            sz = int(rs.integers(1, 3))
            a[y:y + sz, x:x + sz] = v
    else:
        n = int(h * w * 0.0008)
        for _ in range(max(1, n)):
            x = int(rs.uniform(0, w - 1))
            y = int(rs.uniform(0, h * 0.8))
            L = int(rs.uniform(0.03, 0.12) * h)
            th = max(1, int(w * 0.0012))
            y2 = min(h, y + L)
            if y2 > y:
                a[y:y2, x:x + th] = float(rs.uniform(170, 225))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def apply_photometry(img: Image.Image, rng: random.Random) -> tuple[Image.Image, list[str]]:
    """Применяет набор фотоусловий. Возвращает (изображение, список тегов conditions)."""
    conds: list[str] = []
    order = ["night", "rain", "snow", "dirt", "glare", "motion_blur"]
    rng.shuffle(order)
    for c in order:
        if rng.random() > 0.28:
            continue
        if c == "night":
            img = make_night(img, rng)
        elif c == "rain":
            img = add_rain_snow(img, rng, snow=False)
        elif c == "snow":
            img = add_rain_snow(img, rng, snow=True)
        elif c == "dirt":
            img = add_dirt(img, rng)
        elif c == "glare":
            img = add_glare(img, rng)
        elif c == "motion_blur":
            img = add_motion_blur(img, rng)
        conds.append(c)
    if not conds:
        conds.append("day")
    else:
        if "night" not in conds:
            conds.insert(0, "day")
    return img, conds
