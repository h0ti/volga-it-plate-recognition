"""Локальный графический интерфейс для ручной разметки изображений с ГРЗ."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import logging
import re
import sys
import tempfile
import tkinter as tk
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, BinaryIO

from PIL import Image, ImageOps, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "commons" / "images"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "real_review_inbox" / "review.csv"
ERROR_LOG = PROJECT_ROOT / "dataset" / "real_review_inbox" / "annotation_errors.log"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UNKNOWN_LICENSE = "ПРОВЕРИТЬ ЛИЦЕНЗИЮ"
CSV_FIELDS = (
    "annotation_id",
    "image",
    "bbox",
    "plate_num",
    "plate_type",
    "is_vehicle",
    "source_url",
    "license",
    "license_verified",
    "attribution",
    "manual_review",
    "notes",
)
TYPE_LABELS = {
    "type1": "Тип 1 — обычный, одна строка",
    "type1a": "Тип 1А — квадратный, две строки",
    "type1b": "Тип 1Б — жёлтый, одна строка",
    "other": "Другое / нецелевой знак",
}
_FAULT_LOG: BinaryIO | None = None


@dataclass(frozen=True)
class Provenance:
    """Публичные сведения о происхождении изображения из локального manifest."""

    source_url: str = ""
    license: str = UNKNOWN_LICENSE
    attribution: str = ""


def validate_plate_number(plate_num: str, plate_type: str) -> str | None:
    """Возвращает сообщение об ошибке, если номер не соответствует маске типа."""
    value = plate_num.strip().upper().replace(" ", "").replace("-", "")
    letters = "ABEKMHOPCTYX"
    if plate_type in {"type1", "type1a"}:
        pattern = rf"[{letters}][0-9]{{3}}[{letters}]{{2}}[0-9]{{2,3}}"
        example = "A123BC77 или A123BC777"
    elif plate_type == "type1b":
        pattern = rf"[{letters}]{{2}}[0-9]{{3}}[0-9]{{2,3}}"
        example = "AB12377 или AB123777"
    elif plate_type == "other":
        if len(value) > 16 or not re.fullmatch(r"[A-Z0-9#]*", value):
            return "Для «другое» используйте латинские буквы, цифры или #."
        return None
    else:
        return "Сначала выберите тип номера."
    if not re.fullmatch(pattern, value):
        return f"Формат не подходит. Пример: {example}. Допустимые буквы: {letters}."
    return None


def load_provenance(image_dir: Path) -> dict[str, Provenance]:
    """Читает сведения о происхождении из manifest.json рядом с папкой изображений."""
    manifest_path = image_dir / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = image_dir.parent / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать {manifest_path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"В {manifest_path} ожидается JSON-массив.")
    result: dict[str, Provenance] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("local"), str):
            continue
        result[item["local"]] = Provenance(
            source_url=str(item.get("descurl") or item.get("url") or ""),
            license=str(item.get("license") or UNKNOWN_LICENSE),
            attribution=str(item.get("author") or ""),
        )
    return result


def read_annotations(path: Path) -> dict[str, list[dict[str, str]]]:
    """Читает существующую разметку, не теряя ранее сохранённые строки."""
    if not path.is_file():
        return {}
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, delimiter=";")
            if reader.fieldnames is None or not set(CSV_FIELDS).issubset(reader.fieldnames):
                raise ValueError(f"В {path} неожиданная схема CSV.")
            for row in reader:
                by_image[row["image"]].append({key: row.get(key, "") or "" for key in CSV_FIELDS})
    except (OSError, csv.Error) as exc:
        raise ValueError(f"Не удалось прочитать {path}: {exc}") from exc
    return dict(by_image)


def write_annotations(path: Path, rows_by_image: dict[str, list[dict[str, str]]]) -> None:
    """Атомарно записывает все аннотации, сохраняя предыдущие изображения."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for image_rows in rows_by_image.values() for row in image_rows]
    rows.sort(key=lambda row: (row["image"].casefold(), row["annotation_id"]))
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8-sig",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temp_name = output.name
            writer = csv.DictWriter(
                output,
                fieldnames=CSV_FIELDS,
                delimiter=";",
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        Path(temp_name).replace(path)
    except OSError:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
        raise


class AnnotationApp:
    """Оконное приложение для разметки прямоугольников номеров."""

    def __init__(
        self,
        root: tk.Tk,
        image_dir: Path,
        output_path: Path,
        include_reviewed: bool = False,
    ) -> None:
        self.root = root
        self.image_dir = image_dir.resolve()
        self.output_path = output_path.resolve()
        self.root.title("Разметка российских госномеров")
        self.root.geometry("1360x900")
        self.root.minsize(1050, 720)
        self.root.report_callback_exception = self._report_callback_exception
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        try:
            self.provenance = load_provenance(self.image_dir)
            self.rows_by_image = read_annotations(self.output_path)
        except ValueError as exc:
            messagebox.showerror("Не удалось открыть данные", str(exc))
            raise

        self.include_reviewed = tk.BooleanVar(value=include_reviewed)
        self.type_var = tk.StringVar(value="type1")
        self.plate_var = tk.StringVar()
        self.vehicle_var = tk.BooleanVar(value=True)
        self.license_checked_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar()
        self.license_var = tk.StringVar()
        self.source_var = tk.StringVar()
        self.help_var = tk.StringVar()
        self.image_paths: list[Path] = []
        self.image_index = 0
        self.current_image: Image.Image | None = None
        self.display_image: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.display_origin = (0, 0)
        self.drag_start: tuple[int, int] | None = None
        self.pending_box: tuple[int, int, int, int] | None = None
        self.current_boxes: list[tuple[int, int, int, int]] = []
        self.canvas_image_id: int | None = None
        self.pending_rectangle_id: int | None = None
        self.drawn_ids: list[int] = []
        self._render_job: str | None = None
        self._build_ui()
        self._load_image_list()
        self.root.bind("<KeyPress>", self._on_keypress)
        self.root.bind("<Control-s>", lambda _event: self.save_current())
        self.root.bind("<Left>", lambda _event: self.previous_image())
        self.root.bind("<Right>", lambda _event: self.next_image())
        self.root.bind("<Return>", lambda _event: self.add_annotation())

    def _report_callback_exception(
        self,
        exception_type: type[BaseException],
        exception: BaseException,
        traceback: Any,
    ) -> None:
        """Log Tk callback errors and keep the annotation window open."""
        logging.getLogger(__name__).error(
            "Tk callback failed",
            exc_info=(exception_type, exception, traceback),
        )
        self.status_var.set(f"Ошибка записана в {ERROR_LOG}")

    def _close(self) -> None:
        """Save progress before closing the window."""
        try:
            write_annotations(self.output_path, self.rows_by_image)
        except OSError:
            logging.getLogger(__name__).exception("Could not save annotations on close")
            if not messagebox.askyesno(
                "Ошибка сохранения",
                f"Не удалось сохранить разметку. Закрыть всё равно?\n\n{ERROR_LOG}",
            ):
                return
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
            self._render_job = None
        self.root.destroy()

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(
            header,
            text=(
                "Обведи номер мышью → выбери тип и введи символы → "
                "«Добавить разметку». Можно отметить несколько номеров на фото."
            ),
            wraplength=1300,
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Клавиши: 1/2/3/4 — типы · V — на ТС · Enter — добавить · "
            "Ctrl+S — сохранить · ←/→ — фото · N — нет целевого номера",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        image_frame = ttk.Frame(body)
        body.add(image_frame, weight=4)
        self.canvas = tk.Canvas(
            image_frame,
            background="#242424",
            highlightthickness=1,
            highlightbackground="#777777",
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._schedule_render_image)
        self.canvas.bind("<ButtonPress-1>", self._start_box)
        self.canvas.bind("<B1-Motion>", self._drag_box)
        self.canvas.bind("<ButtonRelease-1>", self._finish_box)

        panel = ttk.Frame(body, width=390, padding=(12, 0, 0, 0))
        body.add(panel, weight=2)
        self.counter_label = ttk.Label(panel, text="")
        self.counter_label.pack(anchor="w", pady=(0, 6))
        self.filename_label = ttk.Label(panel, text="", wraplength=370)
        self.filename_label.pack(anchor="w")

        provenance_frame = ttk.LabelFrame(panel, text="Источник и лицензия", padding=8)
        provenance_frame.pack(fill="x", pady=(10, 8))
        ttk.Label(provenance_frame, textvariable=self.source_var, wraplength=350).pack(anchor="w")
        ttk.Label(provenance_frame, textvariable=self.license_var, wraplength=350).pack(
            anchor="w", pady=(4, 0)
        )
        ttk.Checkbutton(
            provenance_frame,
            text="Я отдельно проверил лицензию/разрешение",
            variable=self.license_checked_var,
        ).pack(anchor="w", pady=(5, 0))
        ttk.Button(
            provenance_frame,
            text="Открыть страницу источника",
            command=self.open_source,
        ).pack(anchor="w", pady=(5, 0))

        type_frame = ttk.LabelFrame(panel, text="Тип номера", padding=8)
        type_frame.pack(fill="x", pady=5)
        for key, title in TYPE_LABELS.items():
            row = ttk.Frame(type_frame)
            row.pack(fill="x", anchor="w")
            ttk.Radiobutton(
                row,
                text=f"{key} — {title}",
                variable=self.type_var,
                value=key,
                command=self._update_mask_help,
            ).pack(anchor="w")
        self.help_label = ttk.Label(type_frame, textvariable=self.help_var, wraplength=350)
        self.help_label.pack(anchor="w", pady=(4, 0))

        ttk.Label(panel, text="Номер (латиница; пробелы можно не ставить):").pack(
            anchor="w", pady=(7, 2)
        )
        self.plate_entry = ttk.Entry(panel, textvariable=self.plate_var, font=("Consolas", 16))
        self.plate_entry.pack(fill="x")
        self.plate_entry.bind("<KeyRelease>", lambda _event: self.plate_var.set(
            self.plate_var.get().upper()
        ))
        ttk.Label(
            panel,
            text="Важно: используйте латинские ABEKMHOPCTYX, похожие на кириллицу.",
            wraplength=350,
        ).pack(anchor="w", pady=(3, 0))

        ttk.Checkbutton(
            panel,
            text="Знак установлен на транспортном средстве",
            variable=self.vehicle_var,
            command=lambda: self.root.focus_set(),
        ).pack(anchor="w", pady=(8, 4))

        box_frame = ttk.LabelFrame(panel, text="Выделенная область", padding=8)
        box_frame.pack(fill="x", pady=5)
        self.box_label = ttk.Label(box_frame, text="Перетащи мышью рамку вокруг одного номера.")
        self.box_label.pack(anchor="w")
        ttk.Button(box_frame, text="Убрать последнюю рамку", command=self.remove_last_box).pack(
            anchor="w", pady=(5, 0)
        )

        ttk.Button(
            panel,
            text="Добавить разметку к фото (Enter)",
            command=self.add_annotation,
        ).pack(fill="x", pady=(8, 4))
        ttk.Button(
            panel,
            text="Отметить: целевого номера нет (N)",
            command=self.mark_no_target,
        ).pack(fill="x", pady=4)
        ttk.Button(panel, text="Сохранить (Ctrl+S)", command=self.save_current).pack(
            fill="x", pady=4
        )

        self.annotations_list = tk.Listbox(panel, height=6)
        self.annotations_list.pack(fill="both", expand=False, pady=(6, 2))
        ttk.Button(
            panel,
            text="Удалить выбранную разметку",
            command=self.remove_selected_annotation,
        ).pack(fill="x", pady=(0, 6))

        nav = ttk.Frame(panel)
        nav.pack(fill="x", pady=(3, 0))
        ttk.Button(nav, text="← Назад", command=self.previous_image).pack(
            side="left", fill="x", expand=True, padx=(0, 3)
        )
        ttk.Button(nav, text="Пропустить →", command=self.skip_image).pack(
            side="left", fill="x", expand=True, padx=3
        )
        ttk.Button(nav, text="Далее →", command=self.next_image).pack(
            side="left", fill="x", expand=True, padx=(3, 0)
        )
        ttk.Checkbutton(
            panel,
            text="Показывать уже размеченные фото",
            variable=self.include_reviewed,
            command=self._load_image_list,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Label(panel, textvariable=self.status_var, wraplength=370).pack(
            anchor="w", pady=(6, 0)
        )
        self._update_mask_help()

    def _load_image_list(self) -> None:
        if not self.image_dir.is_dir():
            selected = filedialog.askdirectory(title="Выбери папку с JPG/PNG")
            if not selected:
                self.root.after(100, self.root.destroy)
                return
            self.image_dir = Path(selected).resolve()
            try:
                self.provenance = load_provenance(self.image_dir)
            except ValueError as exc:
                messagebox.showerror("Ошибка manifest", str(exc))
                return
        candidates = sorted(
            (path for path in self.image_dir.rglob("*")
             if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: str(path).casefold(),
        )
        if self.include_reviewed.get():
            self.image_paths = candidates
        else:
            self.image_paths = [
                path for path in candidates
                if not self._is_reviewed(self._relative_name(path))
            ]
        self.image_index = min(self.image_index, max(0, len(self.image_paths) - 1))
        if not self.image_paths:
            self.canvas.delete("all")
            self.canvas_image_id = None
            self.pending_rectangle_id = None
            self.drawn_ids.clear()
            self.display_image = None
            self.status_var.set("Все изображения уже размечены/пропущены.")
            self.counter_label.config(text="Нет неразмеченных изображений")
            self.filename_label.config(text="")
            self._refresh_annotations()
            return
        self._show_image()

    def _relative_name(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.image_dir).as_posix()
        except ValueError:
            return path.name

    def _is_reviewed(self, name: str) -> bool:
        rows = self.rows_by_image.get(name, [])
        return any(row.get("manual_review") in {"annotated", "negative", "skipped"} for row in rows)

    def _current_path(self) -> Path | None:
        if not self.image_paths or not 0 <= self.image_index < len(self.image_paths):
            return None
        return self.image_paths[self.image_index]

    def _show_image(self) -> None:
        path = self._current_path()
        if path is None:
            return
        try:
            with Image.open(path) as source:
                self.current_image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Не удалось открыть изображение", f"{path}\n\n{exc}")
            self.next_image()
            return
        name = self._relative_name(path)
        self.counter_label.config(text=f"Фото {self.image_index + 1} из {len(self.image_paths)}")
        self.filename_label.config(text=name)
        metadata = self.provenance.get(path.name, Provenance())
        self.source_var.set(metadata.source_url or "Источник не найден в manifest")
        self.license_var.set(f"Лицензия в manifest: {metadata.license}")
        self.license_checked_var.set(
            any(row.get("license_verified") == "1" for row in self.rows_by_image.get(name, []))
        )
        rows = self.rows_by_image.get(name, [])
        self.current_boxes = []
        for row in rows:
            if row.get("bbox"):
                try:
                    self.current_boxes.append(tuple(map(int, row["bbox"].split(","))))
                except ValueError:
                    continue
        self.pending_box = None
        self.plate_var.set("")
        self.type_var.set("type1")
        self.vehicle_var.set(True)
        self._update_mask_help()
        self._render_image()
        self._refresh_annotations()
        self.status_var.set("Обведи номер мышью. Если номера нет — нажми N.")

    def _render_image(self) -> None:
        self._render_job = None
        if self.current_image is None:
            return

        for item_id in self.drawn_ids:
            self.canvas.delete(item_id)
        self.drawn_ids.clear()
        if self.pending_rectangle_id is not None:
            self.canvas.delete(self.pending_rectangle_id)
        self.pending_rectangle_id = None
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)
        image_w, image_h = self.current_image.size
        scale = min((canvas_w - 20) / image_w, (canvas_h - 20) / image_h)
        width = max(1, round(image_w * scale))
        height = max(1, round(image_h * scale))
        resized = self.current_image.resize((width, height), Image.Resampling.LANCZOS)
        new_display_image = ImageTk.PhotoImage(resized)
        self.display_scale = scale
        self.display_origin = ((canvas_w - width) // 2, (canvas_h - height) // 2)
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(
                self.display_origin[0],
                self.display_origin[1],
                image=new_display_image,
                anchor="nw",
            )
        else:
            self.canvas.coords(
                self.canvas_image_id,
                self.display_origin[0],
                self.display_origin[1],
            )
            self.canvas.itemconfigure(self.canvas_image_id, image=new_display_image)
        self.display_image = new_display_image
        for box in self.current_boxes:
            self.drawn_ids.append(self._draw_box(box, "#00d26a", 2))
        if self.pending_box:
            self.pending_rectangle_id = self._draw_box(self.pending_box, "#ffcc00", 2)

    def _schedule_render_image(self, _event: tk.Event) -> None:
        """Coalesce resize events to avoid repeatedly recreating Tk image resources."""
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
        self._render_job = self.root.after(75, self._render_image)

    def _canvas_to_original(self, x: int, y: int) -> tuple[int, int]:
        origin_x, origin_y = self.display_origin
        if self.current_image is None:
            return 0, 0
        image_x = round((x - origin_x) / self.display_scale)
        image_y = round((y - origin_y) / self.display_scale)
        return (
            min(max(image_x, 0), self.current_image.width - 1),
            min(max(image_y, 0), self.current_image.height - 1),
        )

    def _draw_box(self, box: tuple[int, int, int, int], color: str, width: int) -> int:
        ox, oy = self.display_origin
        x1, y1, x2, y2 = box
        return self.canvas.create_rectangle(
            ox + x1 * self.display_scale,
            oy + y1 * self.display_scale,
            ox + x2 * self.display_scale,
            oy + y2 * self.display_scale,
            outline=color,
            width=width,
        )

    def _start_box(self, event: tk.Event) -> None:
        if self.current_image is None:
            return
        x, y = self._canvas_to_original(event.x, event.y)
        self.drag_start = (x, y)
        self.pending_box = (x, y, x, y)
        ox, oy = self.display_origin
        px, py = ox + x * self.display_scale, oy + y * self.display_scale
        self.pending_rectangle_id = self.canvas.create_rectangle(
            px, py, px, py, outline="#ffcc00", width=2
        )

    def _drag_box(self, event: tk.Event) -> None:
        if self.drag_start is None or self.pending_rectangle_id is None:
            return
        x, y = self._canvas_to_original(event.x, event.y)
        x0, y0 = self.drag_start
        self.pending_box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        ox, oy = self.display_origin
        x1, y1, x2, y2 = self.pending_box
        self.canvas.coords(
            self.pending_rectangle_id,
            ox + x1 * self.display_scale,
            oy + y1 * self.display_scale,
            ox + x2 * self.display_scale,
            oy + y2 * self.display_scale,
        )

    def _finish_box(self, event: tk.Event) -> None:
        if self.drag_start is None or self.pending_rectangle_id is None:
            return
        x, y = self._canvas_to_original(event.x, event.y)
        x0, y0 = self.drag_start
        box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        self.drag_start = None
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            self.pending_box = None
            self.canvas.delete(self.pending_rectangle_id)
            self.pending_rectangle_id = None
            self.box_label.config(text="Рамка слишком маленькая. Обведи знак целиком.")
        else:
            self.pending_box = box
            ox, oy = self.display_origin
            x1, y1, x2, y2 = box
            self.canvas.coords(
                self.pending_rectangle_id,
                ox + x1 * self.display_scale,
                oy + y1 * self.display_scale,
                ox + x2 * self.display_scale,
                oy + y2 * self.display_scale,
            )
            self.box_label.config(text=f"Рамка: x={box[0]}..{box[2]}, y={box[1]}..{box[3]}")
            self.canvas.itemconfigure(self.pending_rectangle_id, outline="#ffcc00", width=2)

    def _update_mask_help(self) -> None:
        examples = {
            "type1": "Маска: L D D D L L R R(R). Пример: A123BC77",
            "type1a": "Две строки, маска та же: верх A123, низ BC77(7)",
            "type1b": "Жёлтый: L L D D D R R(R). Пример: AB12377",
            "other": "Любой другой знак или нецелевой номер; номер можно оставить пустым.",
        }
        self.help_var.set(examples.get(self.type_var.get(), ""))

    def _on_keypress(self, event: tk.Event) -> None:
        widget = event.widget
        if isinstance(widget, (tk.Entry, ttk.Entry)):
            return
        key = event.keysym.lower()
        key_types = {"1": "type1", "2": "type1a", "3": "type1b", "4": "other"}
        if key in key_types:
            self.type_var.set(key_types[key])
            self._update_mask_help()
        elif key == "v":
            self.vehicle_var.set(not self.vehicle_var.get())
        elif key == "n":
            self.mark_no_target()

    def _current_provenance(self) -> Provenance:
        path = self._current_path()
        return self.provenance.get(path.name, Provenance()) if path else Provenance()

    def add_annotation(self) -> None:
        path = self._current_path()
        if path is None:
            return
        if self.pending_box is None:
            messagebox.showinfo("Нужна рамка", "Сначала обведи номер мышью на изображении.")
            return
        plate_type = self.type_var.get()
        plate_num = self.plate_var.get().strip().upper().replace(" ", "").replace("-", "")
        error = validate_plate_number(plate_num, plate_type)
        if error:
            messagebox.showwarning("Проверь формат номера", error)
            self.plate_entry.focus_set()
            return
        if plate_type in {"type1", "type1a", "type1b"} and not plate_num:
            messagebox.showwarning("Нужен номер", "Для целевого типа введи номер. Нечитаемые символы отметь #.")
            return
        name = self._relative_name(path)
        existing = [
            row for row in self.rows_by_image.get(name, [])
            if row.get("manual_review") not in {"skipped", "negative"}
        ]
        index = len(existing) + 1
        metadata = self._current_provenance()
        box = ",".join(map(str, self.pending_box))
        row = self._make_row(
            name,
            index,
            box,
            plate_num,
            plate_type,
            "1" if self.vehicle_var.get() else "0",
            metadata,
            "1" if self.license_checked_var.get() else "0",
            "annotated",
        )
        existing.append(row)
        self.rows_by_image[name] = existing
        self.current_boxes.append(self.pending_box)
        self.pending_box = None
        self.plate_var.set("")
        self._refresh_annotations()
        self._render_image()
        self.save_current()
        self.status_var.set(f"Сохранено разметок на этом фото: {len(existing)}.")

    def _make_row(
        self,
        name: str,
        index: int,
        box: str,
        plate_num: str,
        plate_type: str,
        is_vehicle: str,
        metadata: Provenance,
        license_verified: str,
        status: str,
    ) -> dict[str, str]:
        return {
            "annotation_id": f"{name}#{index:02d}",
            "image": name,
            "bbox": box,
            "plate_num": plate_num,
            "plate_type": plate_type,
            "is_vehicle": is_vehicle,
            "source_url": metadata.source_url,
            "license": metadata.license,
            "license_verified": license_verified,
            "attribution": metadata.attribution,
            "manual_review": status,
            "notes": "",
        }

    def mark_no_target(self) -> None:
        path = self._current_path()
        if path is None:
            return
        name = self._relative_name(path)
        if not messagebox.askyesno(
            "Подтвердить",
            "На фото действительно нет целевого номера (type1/1a/1b)?",
        ):
            return
        metadata = self._current_provenance()
        self.rows_by_image[name] = [
            self._make_row(
                name,
                0,
                "",
                "",
                "other",
                "0",
                metadata,
                "1" if self.license_checked_var.get() else "0",
                "negative",
            )
        ]
        self.save_current()
        self.next_image()

    def _refresh_annotations(self) -> None:
        self.annotations_list.delete(0, tk.END)
        path = self._current_path()
        if path is None:
            return
        rows = self.rows_by_image.get(self._relative_name(path), [])
        for row in rows:
            if row["manual_review"] == "skipped":
                text = "Пропущено"
            elif row["manual_review"] == "negative":
                text = "Нет целевого номера"
            else:
                text = (
                    f"{row['plate_type']}  {row['plate_num']}  "
                    f"ТС={row['is_vehicle']}  box={row['bbox']}"
                )
            self.annotations_list.insert(tk.END, text)

    def remove_last_box(self) -> None:
        if self.pending_box:
            self.pending_box = None
        elif self.current_boxes:
            self.current_boxes.pop()
            path = self._current_path()
            if path:
                name = self._relative_name(path)
                rows = self.rows_by_image.get(name, [])
                annotated = [row for row in rows if row["manual_review"] == "annotated"]
                if annotated:
                    annotated.pop()
                    other = [row for row in rows if row["manual_review"] != "annotated"]
                    self.rows_by_image[name] = other + annotated
        self._render_image()
        self._refresh_annotations()

    def remove_selected_annotation(self) -> None:
        selected = self.annotations_list.curselection()
        path = self._current_path()
        if not selected or path is None:
            return
        name = self._relative_name(path)
        rows = self.rows_by_image.get(name, [])
        index = selected[0]
        if 0 <= index < len(rows):
            rows.pop(index)
            self.rows_by_image[name] = rows
            self.current_boxes = [
                tuple(map(int, row["bbox"].split(",")))
                for row in rows
                if row.get("bbox") and row.get("manual_review") == "annotated"
            ]
            self._refresh_annotations()
            self._render_image()
            self.save_current()

    def save_current(self) -> None:
        path = self._current_path()
        if path is None:
            return
        try:
            write_annotations(self.output_path, self.rows_by_image)
        except OSError as exc:
            messagebox.showerror("Ошибка сохранения", f"{self.output_path}\n\n{exc}")
            return
        self.status_var.set(f"Сохранено: {self.output_path}")

    def skip_image(self) -> None:
        path = self._current_path()
        if path is None:
            return
        name = self._relative_name(path)
        metadata = self._current_provenance()
        self.rows_by_image[name] = [
            self._make_row(
                name,
                0,
                "",
                "",
                "",
                "",
                metadata,
                "1" if self.license_checked_var.get() else "0",
                "skipped",
            )
        ]
        self.save_current()
        self.next_image()

    def next_image(self) -> None:
        if self._current_path() is None:
            return
        name = self._relative_name(self._current_path())
        if name in self.rows_by_image:
            self.save_current()
        self.image_index += 1
        if self.image_index >= len(self.image_paths):
            self.image_index = len(self.image_paths) - 1
            self.status_var.set("Достигнут конец списка. Разметка сохранена.")
            return
        self._show_image()

    def previous_image(self) -> None:
        if self.image_paths and self.image_index > 0:
            name = self._relative_name(self._current_path())
            if name in self.rows_by_image:
                self.save_current()
            self.image_index -= 1
            self._show_image()

    def open_source(self) -> None:
        source_url = self._current_provenance().source_url
        if not source_url:
            messagebox.showinfo("Источник не найден", "Для этого файла нет ссылки в manifest.")
            return
        webbrowser.open(source_url)


def build_parser() -> argparse.ArgumentParser:
    """Создаёт аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Оконная ручная разметка фотографий с номерами.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-reviewed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Запускает оконное приложение."""
    global _FAULT_LOG
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=ERROR_LOG,
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
    )
    _FAULT_LOG = ERROR_LOG.open("ab", buffering=0)
    faulthandler.enable(file=_FAULT_LOG, all_threads=True)
    args = build_parser().parse_args(argv)
    try:
        root = tk.Tk()
        AnnotationApp(root, args.images, args.output, args.include_reviewed)
        root.mainloop()
    except tk.TclError as exc:
        print(f"Не удалось запустить графическое окно: {exc}", file=sys.stderr)
        return 1
    except Exception:
        logging.getLogger(__name__).exception("Fatal annotation application error")
        print(f"Ошибка разметчика. Подробности: {ERROR_LOG}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
