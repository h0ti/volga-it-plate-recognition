"""Сборка датасета в структуре раздела 6 задания.

    dataset/
      images/{real,synthetic}/   meta.csv   labels/<stem>.txt
      generator/                 README.md  LICENSE

Ключевой момент: синтетика генерируется детерминированно (--seed), поэтому
набор полностью воспроизводим, а meta.csv и labels/ пишутся из тех же данных,
что и картинки, — расхождений быть не может.

Запуск:
    python -m tools.build_dataset --out dataset --synthetic 6000 --seed 20260903
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys
from typing import Any

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platekit import render, scene, spec  # noqa: E402

SYNTH_SOURCE = "own_synthetic"
SYNTH_LICENSE = "ПРОВЕРИТЬ ЛИЦЕНЗИЮ"
SEED_DEFAULT = 20260903

# Распределение типов в синтетике. «other» нужен детектору и классификатору типа,
# чтобы отличать знаки от прочих прямоугольников; реальные негативы — отдельные строки.
SYNTH_MIX = (("type1a", 0.30), ("type1", 0.26), ("type1b", 0.24), ("other", 0.20))


# ---------------------------------------------------------------------------
# Отрисовка одного синтетического изображения
# ---------------------------------------------------------------------------

def _fake_plate_text(rng: random.Random) -> str:
    """Мусорный «номер» для класса other: буквы/цифры, которых нет в маске."""
    alphabet = spec.LATIN_LETTERS + "0123456789"
    n = rng.randint(5, 11)
    return "".join(rng.choice(alphabet) for _ in range(n))


def make_other_plate(rng: random.Random, text: str, width: int = 520) -> Image.Image:
    """Знак, похожий на номер, но не являющийся таковым.

    Это ключевая часть требования п. 3.7: модель обязана отличать ГРЗ от
    рекламных щитов, витрин, наклеек и прочих прямоугольных вывесок.
    """
    h = max(10, int(round(width * rng.uniform(0.16, 0.62))))
    bg = rng.choice([(250, 250, 248), (250, 250, 248), (238, 240, 244),
                     (252, 240, 200), (232, 236, 232)])
    im = Image.new("RGB", (width, h), bg)
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(im)
    fonts = render.available_fonts()
    if fonts and rng.random() < 0.85:
        fs = max(6, int(h * rng.uniform(0.45, 0.8)))
        fnt = render.get_font(rng.choice(fonts), fs)
        d.text((int(rng.uniform(2, max(3, width * 0.1))), int(h * 0.12)), text,
               font=fnt, fill=(rng.randint(20, 90),) * 3)
    else:
        for _ in range(rng.randint(2, 7)):        # полосы/блоки без читаемого текста
            x = rng.uniform(0, width * 0.9)
            y = rng.uniform(0, h * 0.8)
            ww = rng.uniform(width * 0.1, width * 0.7)
            d.rectangle([x, y, x + ww, y + rng.uniform(2, h * 0.5)],
                        fill=tuple(rng.randint(0, 255) for _ in range(3)))
    d.rectangle([0, 0, width - 1, h - 1],
                outline=(rng.randint(80, 190),) * 3, width=max(1, width // 200))
    return im


def make_synthetic_image(rng: random.Random, pool: scene.BackgroundPool,
                         idx: int, out_img: str) -> list[dict[str, Any]]:
    """Один синтетический кадр + список знаков (строки meta.csv)."""
    W = rng.choice([(640, 480), (720, 540), (800, 600), (960, 720), (1024, 768)])
    canvas = pool.pick(W)

    r = rng.random()
    acc = 0.0
    ptype = SYNTH_MIX[-1][0]
    for name, p in SYNTH_MIX:
        acc += p
        if r <= acc:
            ptype = name
            break

    n_plates = 1 if rng.random() < 0.86 else rng.randint(2, 3)
    rows: list[dict[str, Any]] = []
    for k in range(n_plates):
        if ptype == "other":
            fake_text = _fake_plate_text(rng)
            pim = make_other_plate(rng, fake_text)
            plate = ""
        else:
            plate = spec.random_plate(rng, ptype)
            pim = render.render_plate(
                plate, ptype, 520, rng,
                draw_flag=(ptype == "type1a" and rng.random() < 0.45),
            )
        center = scene.sample_center(rng, W[0], W[1], 0.2)
        # Однострочные знаки (1, 1Б) уже вытянуты (4.6:1), поэтому сильная
        # перспектива схлопывает их в квадрат — валидатор на это ругается.
        # У двухстрочного 1А (1.7:1) перспектива безопасна.
        strength = rng.uniform(0.15, 0.55) if ptype in ("type1", "type1b") \
            else rng.uniform(0.30, 1.00)
        quad, _ = scene.paste_plate_on_canvas(canvas, pim, center, rng, strength)
        if not quad:
            continue
        quad = scene.ensure_clockwise(quad)
        x, y, w, h = scene.bbox_of_quad(quad)
        # Валидатор: bbox внутри кадра, не мелкий.
        if w < 14 or h < 8 or x + w >= W[0] or y + h >= W[1]:
            continue
        rows.append({
            "plate": plate,
            "plate_type": ptype,
            "bbox": (x, y, w, h),
            "quad": quad,
            "is_vehicle": 0 if ptype == "other" else 1,
        })

    if not rows:
        return []
    canvas, conds = render.apply_photometry(canvas, rng)
    if rng.random() < 0.35 and "angle" not in conds:
        conds.append("angle")
    canvas.save(out_img, quality=rng.randint(72, 95))

    out = []
    stem = os.path.splitext(os.path.basename(out_img))[0]
    for row in rows:
        x, y, w, h = row["bbox"]
        out.append({
            "image": f"images/synthetic/{stem}.jpg",
            "plate_num": row["plate"],
            "plate_type": row["plate_type"],
            "bbox": f"{x},{y},{w},{h}",
            "quad": ",".join(str(int(round(px))) for p in row["quad"]
                             for px in p),
            "is_vehicle": row["is_vehicle"],
            "is_synthetic": 1,
            "source": SYNTH_SOURCE,
            "license": SYNTH_LICENSE,
            "conditions": ",".join(conds),
        })
    return out


def make_negative_image(rng: random.Random, pool: scene.BackgroundPool, out_img: str) -> None:
    """Write a background-only frame for detector objectness negatives."""
    size = rng.choice([(640, 480), (720, 540), (800, 600), (960, 720), (1024, 768)])
    canvas = pool.pick(size)
    canvas, _ = render.apply_photometry(canvas, rng)
    canvas.save(out_img, quality=rng.randint(72, 95))


# ---------------------------------------------------------------------------
# Запись датасета
# ---------------------------------------------------------------------------

def write_labels(rows: list[dict[str, Any]], labels_dir: str, root: str) -> None:
    """labels/<stem>.txt в формате YOLO: class cx cy w h + 8 чисел quad.

    Нормализация — от размеров изображения, а не от bbox: так файл меток
    гарантированно совпадает с тем, что ожидает validate_dataset.py.
    """
    by_img: dict[str, list[dict]] = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    labeled_stems = {
        os.path.splitext(os.path.basename(img))[0] for img in by_img
    }
    for img, rs in by_img.items():
        stem = os.path.splitext(os.path.basename(img))[0]
        try:
            with Image.open(os.path.join(root, img)) as im:
                iw, ih = im.size
        except OSError as exc:
            raise RuntimeError(f"Не удалось прочитать изображение для разметки: {img}") from exc
        lines = []
        for r in rs:
            x, y, w, h = (int(v) for v in r["bbox"].split(","))
            cls = spec.LABEL_CLASSES[r["plate_type"]]
            cx, cy = (x + w / 2) / iw, (y + h / 2) / ih
            nw, nh = w / iw, h / ih
            q = [int(v) for v in r["quad"].split(",")]
            nq = [f"{q[i] / iw:.6f}" if i % 2 == 0 else f"{q[i] / ih:.6f}"
                  for i in range(8)]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(nq))
        with open(os.path.join(labels_dir, f"{stem}.txt"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    image_dir = os.path.join(root, "images", "synthetic")
    for image_path in glob.glob(os.path.join(image_dir, "*")):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        if stem not in labeled_stems:
            with open(
                os.path.join(labels_dir, f"{stem}.txt"), "w", encoding="utf-8"
            ):
                pass


def write_meta(rows: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(spec.META_COLUMNS)
        for r in rows:
            w.writerow([r[c] for c in spec.META_COLUMNS])


README_TMPL = """# Датасет нестандартных ГРЗ — Волга-IT 2026, полуфинал

Датасет предназначен для обучения и оценки распознавания **нестандартных**
российских государственных регистрационных знаков: тип **1А** (квадратный
двухстрочный, 290×170 мм) и тип **1Б** (жёлтый, 520×112 мм), плюс базовый
тип **1** (520×112 мм) и класс **other** — знаки вне области задания.

Лицензия на весь набор: **CC BY 4.0** (см. `LICENSE`).

## Состав

%(TABLE)s

## Структура

```
dataset/
  images/synthetic/   синтетические кадры (скрипт-генератор, фиксированный seed)
  images/real/        реальные фотографии из открытых источников
  labels/<stem>.txt   разметка YOLO: class cx cy w h + 8 чисел quad
  meta.csv            метаданные, одна строка на один знак (разделитель «;»)
  generator/          скрипт-генератор синтетики + requirements.txt
  README.md           этот файл
  LICENSE             CC BY 4.0
```

## Источники и лицензии

Все включённые материалы распространяются под **CC0**, **Public domain** либо
**CC BY**, то есть совместимо с публикацией набора под CC BY 4.0.
Материалы **CC BY-SA** (запрещает перелицензирование) и **CC BY-ND**
(запрещает производные — кропы и выпрямление перспективы) **в набор не
включались**. Для каждого реального изображения поля `source` и `license`
в `meta.csv` заполнены построчно.

%(SOURCES)s

Полный перечень: %(TOTAL_IMG)d изображений, %(TOTAL_PLATE)d знаков.

## Методика разметки

* Реальные изображения размечены **вручную**: область знака (bbox и quad),
  канонический номер, тип знака и признак «знак находится на транспортном
  средстве». Нечитаемые человеком позиции обозначены `#`.
  Автоматическая разметка посторонней моделью, поданная как ручная, не
  использовалась.
* Синтетические изображения размечены точно: bbox и quad вычисляются из
  параметров вставки, канонический номер известен по построению, разметка
  не может разойтись с картинкой.
* `labels/<stem>.txt` и `meta.csv` генерируются из одних и тех же данных,
  поэтому расхождения между ними невозможны.

## Синтетические данные и воспроизводимость

Генератор: `generator/build_dataset.py`. Набор **полностью воспроизводим** —
зафиксирован random seed (`%(SEED)s`), внешние ассеты и сеть не используются.

Знаки рисуются по геометрии ГОСТ Р 50577-2018 (типы 1 и 1Б — 520×112 мм,
тип 1А — 290×170 мм, высота символа 50 мм), затем вставляются в реальные
фотографии-сцены со случайной перспективой, а затем применяются
фотографические условия, перечисленные в поле `conditions`: `day`, `night`,
`rain`, `snow`, `dirt`, `glare`, `motion_blur`, `angle`.

Синтетика решает основную проблему задания: нестандартные знаки в открытых
наборах практически отсутствуют, и без их генерации модель на типах 1А и 1Б
обучаться не на чем.

## Форматы

**Номер.** Латиница, верхний регистр. Маски по типам:

| тип | маска | пример |
|---|---|---|
| type1, type1a | `Л ЦЦЦ ЛЛ РР(Р)` | `A123BC777` |
| type1b | `ЛЛ ЦЦЦ РР(Р)` | `AB12377` |
| other | `A-Z`, `0-9`, `#`, до 12 символов | пусто |

Буквы: `A B E K M H O P C T Y X`. `#` — позиция, которую человек прочитать не смог.

**`quad`** — четыре угловые точки `x1,y1,...,x4,y4` в пикселях, по часовой
стрелке начиная с левого верхнего угла. **`bbox`** — `x,y,w,h` в пикселях и
согласован с `quad`.

## Ограничения и известные проблемы

* Реальных изображений типа **1Б** (жёлтые знаки) и **1А** (квадратные) в
  открытых источниках с приемлемой лицензией крайне мало — это известная
  трудность сбора. Объёмы по типам указаны в таблице состава.
* Лица людей на реальных изображениях размыты; кадры, где человек является
  смысловым центром композиции, исключены.
* Изображения из отладочного набора организаторов в набор не включены.
"""


def write_readme(rows: list[dict], root: str, seed: int) -> None:
    """README генерируется из meta.csv — статистика всегда соответствует данным."""
    from collections import Counter
    img_by = {}
    for r in rows:
        img_by.setdefault(r["image"], []).append(r)
    real_imgs: dict[str, set] = {}
    real_plates: dict[str, set] = {}
    cond = Counter()
    src = Counter()
    syn_imgs = set()
    for r in rows:
        cond.update(t for t in r["conditions"].split(",") if t)
        src[r["license"]] += 1
        if r["is_synthetic"] == "1":
            syn_imgs.add(r["image"])
        else:
            real_imgs.setdefault(r["plate_type"], set()).add(r["image"])
            if r["plate_num"] and "#" not in r["plate_num"]:
                real_plates.setdefault(r["plate_type"], set()).add(r["plate_num"])

    t = spec.TYPES
    table = ["| Группа | Изображений | Уникальных ГРЗ | Требование п. 6.1 |",
             "|---|---|---|---|"]
    table.append(f"| synthetic (все типы) | {len(syn_imgs)} | — | ≥ 5000 |")
    for name, req_i, req_p in (("type1a", 150, 50), ("type1b", 300, 100),
                               ("other", 50, "—")):
        ni = len(real_imgs.get(name, ()))
        np_ = len(real_plates.get(name, ()))
        need = f"≥ {req_i} фото" + (f" / ≥ {req_p} номеров" if req_p != "—" else "")
        table.append(f"| real: {name} | {ni} | {np_ if req_p != '—' else '—'} | {need} |")
    table.append(f"| **всего** | **{len(img_by)}** | — | — |")

    sources = "| Лицензия | Знаков |\n|---|---|\n" + "\n".join(
        f"| `{k}` | {v} |" for k, v in src.most_common())
    sources += ("\n\nУсловия съёмки: "
                + ", ".join(f"`{k}` — {v}" for k, v in cond.most_common()))

    text = README_TMPL % {
        "TABLE": "\n".join(table),
        "SOURCES": sources,
        "TOTAL_IMG": len(img_by),
        "TOTAL_PLATE": len(rows),
        "SEED": seed,
    }
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(text)


def main(argv: list[str] | None = None) -> int:
    """Генерирует отдельную синтетическую выборку, не перезаписывая dataset/meta.csv."""
    ap = argparse.ArgumentParser(
        description="Генерация синтетических ГРЗ в отдельный каталог."
    )
    ap.add_argument("--out", default="dataset/generated")
    ap.add_argument("--synthetic", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--bg-dir", default="")
    ap.add_argument("--bg-max", type=int, default=400)
    ap.add_argument(
        "--negative-fraction",
        type=float,
        default=0.10,
        help="Share of background-only frames for detector training (0..0.4).",
    )
    args = ap.parse_args(argv)

    if args.synthetic < 1:
        ap.error("--synthetic должен быть больше нуля.")
    if args.bg_max < 1:
        ap.error("--bg-max должен быть больше нуля.")
    if not 0.0 <= args.negative_fraction <= 0.4:
        ap.error("--negative-fraction должен быть от 0 до 0.4.")

    root = os.path.abspath(args.out)
    os.makedirs(root, exist_ok=True)
    if next(os.scandir(root), None) is not None:
        ap.error(f"Каталог вывода не пустой; данные не перезаписаны: {root}")

    syn_dir = os.path.join(root, "images", "synthetic")
    labels_dir = os.path.join(root, "labels")
    for d in (syn_dir, labels_dir):
        os.makedirs(d, exist_ok=True)

    rng = random.Random(args.seed)
    bgs = sorted(glob.glob(os.path.join(args.bg_dir, "*.jpg"))) if args.bg_dir else []
    random.Random(args.seed).shuffle(bgs)
    bgs = bgs[: args.bg_max]
    pool = scene.BackgroundPool(bgs, rng)
    print(f"Фонов: {len(pool)} (реальных), целевых синтетических кадров: {args.synthetic}")

    rows: list[dict] = []
    made = 0
    attempts = 0
    while made < args.synthetic and attempts < args.synthetic * 3:
        attempts += 1
        stem = f"syn_{made:06d}"
        path = os.path.join(syn_dir, f"{stem}.jpg")
        if rng.random() < args.negative_fraction:
            make_negative_image(rng, pool, path)
            made += 1
            continue
        got = make_synthetic_image(rng, pool, made, path)
        if not got:
            if os.path.exists(path):
                os.remove(path)
            continue
        rows.extend(got)
        made += 1
        if made % 500 == 0:
            print(f"  {made}/{args.synthetic}…", flush=True)

    print(f"Синтетических изображений: {made}, знаков: {len(rows)}")
    if made != args.synthetic:
        print(
            f"Ошибка: удалось создать {made} из {args.synthetic} изображений.",
            file=sys.stderr,
        )
        return 1
    write_labels(rows, labels_dir, root)
    write_meta(rows, os.path.join(root, "meta.csv"))
    print(f"meta.csv: {len(rows)} строк")

    stats = {}
    for r in rows:
        stats[r["plate_type"]] = stats.get(r["plate_type"], 0) + 1
    print("По типам:", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
