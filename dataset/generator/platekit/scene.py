"""Композитинг: собирает синтетический кадр с номерами и точной разметкой.

Используется в двух режимах:
  * detector-режим — знак вставляется в кадр произвольного размера (bbox + quad точные),
    на любом фоне, включая негативы «знак не на ТС»;
  * ocr-режим — знак выпрямляется по quad в прямоугольник фиксированной высоты,
    метка равна каноническому номеру.
"""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw, ImageFilter

from . import render, spec


# ---------------------------------------------------------------------------
# Фоны
# ---------------------------------------------------------------------------

def make_background(w: int, h: int, rng: random.Random) -> Image.Image:
    """Процедурный фон: небо/асфальт/стена/трава — грубые, но разнообразные текстуры."""
    import numpy as np
    rs = np.random.default_rng(rng.randint(0, 2**31 - 1))
    kind = rng.choice(("asphalt", "sky", "wall", "grass", "night", "gradient"))

    if kind == "sky":
        top = rs.integers(90, 190, 3)
        bot = rs.integers(140, 235, 3)
        grad = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        a = (top[None, None, :] * (1 - grad) + bot[None, None, :] * grad)
        a = np.repeat(a, w, axis=1)
    elif kind == "asphalt":
        base = rs.integers(35, 95)
        a = np.full((h, w, 3), base, np.float32)
        a += rs.normal(0, 14, (h, w, 1))
        a += rs.normal(0, 6, (1, w, 3))
    elif kind == "wall":
        base = rs.integers(110, 215, 3).astype(np.float32)
        a = np.repeat(base[None, None, :], h, 0).repeat(w, 1).astype(np.float32)
        a += rs.normal(0, 9, (h, w, 3))
        for _ in range(rng.randint(0, 4)):        # кладка / стыки панелей
            y = int(rng.uniform(0, h))
            a[y:y + rng.randint(1, 4), :] *= 0.85
    elif kind == "grass":
        base = rs.integers(45, 115, 3).astype(np.float32)
        a = np.repeat(base[None, None, :], h, 0).repeat(w, 1)
        a += rs.normal(0, 20, (h, w, 3))
    elif kind == "night":
        base = rs.integers(8, 30, 3).astype(np.float32)
        a = np.repeat(base[None, None, :], h, 0).repeat(w, 1)
        a += rs.normal(0, 4, (h, w, 3))
    else:
        g = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
        a = np.repeat(g, h, 0)
        a = a + rs.normal(0, 8, (h, w, 3))

    a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def load_backgrounds(paths: list[str], rng: random.Random,
                     max_side: int = 1600) -> list[Image.Image]:
    """Загружает реальные фотографии как фон (для разнообразия домена)."""
    out: list[Image.Image] = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        if max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))),
                           Image.LANCZOS)
        if min(im.size) >= 200:
            out.append(im)
    return out


class BackgroundPool:
    """Пул фонов: реальные фотографии (основной режим) + процедурные (запасной).

    Реальные сцены принципиальны для детектора: на однородном шуме модель
    выучит «светлый прямоугольник на любом фоне» и провалится на кадрах с камер.
    """

    def __init__(self, real_paths: list[str] | None = None, rng: random.Random | None = None,
                 min_side: int = 480):
        self.rng = rng or random.Random(0)
        self.real: list[Image.Image] = []
        self.min_side = min_side
        if real_paths:
            self.real = load_backgrounds(real_paths, self.rng)
        if not self.real:
            self.real = []   # работаем на процедурных

    def __len__(self) -> int:
        return len(self.real)

    def pick(self, target_size: tuple[int, int]) -> Image.Image:
        if self.real and self.rng.random() < 0.85:
            base = self.rng.choice(self.real)
            W, H = target_size
            # Кадрируем случайный прямоугольник под нужный размер.
            s = min(target_size[0] / base.size[0], target_size[1] / base.size[1])
            s *= self.rng.uniform(0.55, 1.0)
            cw = max(64, int(target_size[0] / s))
            ch = max(64, int(target_size[1] / s))
            cw, ch = min(cw, base.size[0]), min(ch, base.size[1])
            x = self.rng.randint(0, base.size[0] - cw)
            y = self.rng.randint(0, base.size[1] - ch)
            im = base.crop((x, y, x + cw, y + ch)).resize(target_size, Image.LANCZOS)
            if self.rng.random() < 0.25:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            return im
        return make_background(target_size[0], target_size[1], self.rng)


# ---------------------------------------------------------------------------
# Вставка знака в кадр
# ---------------------------------------------------------------------------

def _quad_from_corners(c0, c1, c2, c3) -> list[tuple[float, float]]:
    return [c0, c1, c2, c3]


def paste_plate_on_canvas(
    canvas: Image.Image,
    plate_img: Image.Image,
    center: tuple[float, float],
    rng: random.Random,
    strength: float = 1.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Вставляет знак в canvasPerspective-искажением. Возвращает (quad_canvas, quad_plate_src).

    quad_canvas — 4 точки по часовой стрелке от левого верхнего (то, что идёт в meta.csv);
    quad_plate_src — соответствующие точки на plate_img (для последующего выпрямления).
    """
    pw, ph = plate_img.size
    # Размер вставки: доля от меньшей стороны кадра, с большим разбросом (дальние знаки).
    base = min(canvas.size)
    frac = rng.uniform(0.055, 0.42) ** 1.15
    scale = base * frac / max(pw, 1)
    tw, th = max(12, int(pw * scale)), max(8, int(ph * scale))
    # Знак должен целиком помещаться в кадр: иначе bbox вылезет за границы,
    # а валидатор организаторов считает это ERROR.
    tw = min(tw, int(canvas.size[0] * 0.92))
    th = min(th, int(canvas.size[1] * 0.92))

    warped, src_quad = render.random_perspective(
        plate_img.resize((tw, th), Image.LANCZOS), rng, strength
    )
    W, H = canvas.size
    # Центр ограничиваем так, чтобы вставка гарантированно была внутри кадра.
    cx = min(max(center[0], tw * 0.5), W - tw * 0.5)
    cy = min(max(center[1], th * 0.5), H - th * 0.5)
    x0 = int(round(cx - tw / 2))
    y0 = int(round(cy - th / 2))

    left = max(0, -x0)
    top = max(0, -y0)
    right = min(tw, W - x0)
    bottom = min(th, H - y0)
    if right - left < 12 or bottom - top < 8:
        return [], []
    warped = warped.crop((left, top, right, bottom))

    # Оттенок знака подгоняем под освещение сцены (иначе выглядит наклейкой).
    import numpy as np
    a = np.asarray(warped).astype(np.float32)
    ca = np.asarray(canvas.crop((x0 + left, y0 + top, x0 + right, y0 + bottom))).astype(np.float32)
    shift = float(np.mean(ca) - np.mean(a))
    a = np.clip(a + shift * rng.uniform(0.25, 0.85), 0, 255)
    warped = Image.fromarray(a.astype(np.uint8))

    mask = Image.new("L", warped.size, 0)
    md = ImageDraw.Draw(mask)
    md.rectangle([1, 1, warped.size[0] - 2, warped.size[1] - 2], fill=255)
    if rng.random() < 0.5:
        mask = mask.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.4)))

    canvas.paste(warped, (x0 + left, y0 + top), mask)

    # Пересчитываем quad после обрезки и зажима в границы кадра.
    # src_quad уже задан в ПИКСЕЛЯХ искажённого изображения размера (tw, th),
    # поэтому переносим его в canvas простым сдвигом — умножать на tw/th нельзя.
    ox, oy = x0 + left, y0 + top
    quad_canvas = [(ox + p[0], oy + p[1]) for p in src_quad]
    # Нормализованные координаты на исходном изображении знака — пригодятся,
    # если кроп понадобится пересчитать без повторной выборки.
    quad_src = [(p[0] / max(tw, 1), p[1] / max(th, 1)) for p in src_quad]
    quad_canvas = [
        (min(max(px, 0), W - 1), min(max(py, 0), H - 1)) for px, py in quad_canvas
    ]
    return quad_canvas, quad_src


def bbox_of_quad(quad: list[tuple[float, float]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x0, y0 = int(round(min(xs))), int(round(min(ys)))
    x1, y1 = int(round(max(xs))), int(round(max(ys)))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def ensure_clockwise(quad: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Возвращает quad по часовой стрелке, начиная с левого верхнего угла.

    Требование валидатора организаторов: в экранных координатах (y вниз)
    знаковая площадь положительна, а первая точка — с минимальной суммой x+y.
    Проверено на эталоне TL→TR→BR→BL: площадь = +1, значит угол atan2
    должен ВОЗРАСТАТЬ (на TL он минимален).
    """
    pts = list(quad)
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    start = min(range(4), key=lambda k: pts[k][0] + pts[k][1])
    return pts[start:] + pts[:start]


def sample_center(rng: random.Random, w: int, h: int, plate_wh: float) -> tuple[float, float]:
    """Центр вставки: чаще ближе к середине кадра, иногда у самого края."""
    if rng.random() < 0.72:
        cx = rng.uniform(0.32, 0.68) * w
        cy = rng.uniform(0.38, 0.72) * h
    else:
        cx = rng.uniform(0.06, 0.94) * w
        cy = rng.uniform(0.12, 0.90) * h
    return cx, cy
