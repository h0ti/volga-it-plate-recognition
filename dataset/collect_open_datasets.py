"""Сбор изображений из явно настроенных открытых источников."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps, UnidentifiedImageError


DATASET_DIR = Path(__file__).resolve().parent
COLLECTION_FIELDS = (
    "image",
    "candidate_type",
    "suggestion_method",
    "manual_review",
    "source",
    "source_url",
    "license",
    "license_verified",
    "attribution",
    "status",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
MAX_UNPACKED_BYTES = 10 * 1024 * 1024 * 1024
UNKNOWN_LICENSE = "ПРОВЕРИТЬ ЛИЦЕНЗИЮ"
SOURCE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


@dataclass(frozen=True)
class Source:
    """Настройки одного источника; лицензию указывает и проверяет пользователь."""

    name: str
    provider: str
    source_url: str
    license: str
    license_verified: bool
    attribution: str
    images_are_plate_crops: bool
    urls: tuple[str, ...] = ()
    dataset: str = ""
    workspace: str = ""
    project: str = ""
    version: int = 0
    export_format: str = ""


def _is_http_url(value: str) -> bool:
    """Проверяет, что URL имеет HTTP(S)-схему и заданный хост."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def load_sources(path: Path) -> list[Source]:
    """Загружает и проверяет JSON-конфигурацию источников."""
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать конфигурацию {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise ValueError("В JSON ожидается объект с массивом 'sources'.")

    sources: list[Source] = []
    names: set[str] = set()
    for index, item in enumerate(raw["sources"]):
        if not isinstance(item, dict):
            raise ValueError(f"sources[{index}] должен быть объектом.")
        name = item.get("name")
        provider = item.get("provider")
        source_url = item.get("source_url")
        license_name = item.get("license", UNKNOWN_LICENSE)
        if not isinstance(name, str) or not SOURCE_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"sources[{index}].name должен быть безопасным slug.")
        if name in names:
            raise ValueError(f"Повторяется имя источника: {name}")
        names.add(name)
        if provider not in {"requests", "kaggle", "roboflow"}:
            raise ValueError(f"{name}: provider должен быть requests, kaggle или roboflow.")
        if not isinstance(source_url, str) or not _is_http_url(source_url):
            raise ValueError(f"{name}: укажите HTTP(S) source_url с подтверждённой страницей источника.")
        if not isinstance(license_name, str) or not license_name.strip():
            raise ValueError(f"{name}: поле license не может быть пустым.")

        verified = item.get("license_verified", False)
        crops = item.get("images_are_plate_crops", False)
        if not isinstance(verified, bool) or not isinstance(crops, bool):
            raise ValueError(f"{name}: license_verified и images_are_plate_crops должны быть boolean.")
        if verified and UNKNOWN_LICENSE.casefold() in license_name.casefold():
            raise ValueError(f"{name}: неизвестную лицензию нельзя пометить проверенной.")

        urls_value = item.get("urls", [])
        if not isinstance(urls_value, list) or any(
            not isinstance(url, str) or not _is_http_url(url) for url in urls_value
        ):
            raise ValueError(f"{name}: urls должен содержать только HTTP(S) URL.")

        dataset = item.get("dataset", "")
        workspace = item.get("workspace", "")
        project = item.get("project", "")
        export_format = item.get("format", "")
        version = item.get("version", 0)
        if provider == "requests" and not urls_value:
            raise ValueError(f"{name}: для requests требуется непустой массив urls.")
        if provider == "kaggle" and (
            not isinstance(dataset, str) or not re.fullmatch(r"[^/\s]+/[^/\s]+", dataset)
        ):
            raise ValueError(f"{name}: dataset должен иметь формат owner/dataset.")
        if provider == "roboflow":
            if not all(isinstance(value, str) and value for value in (workspace, project, export_format)):
                raise ValueError(f"{name}: для Roboflow требуются workspace, project и format.")
            if not isinstance(version, int) or isinstance(version, bool) or version < 1:
                raise ValueError(f"{name}: version должен быть положительным целым числом.")

        attribution = item.get("attribution", "")
        if not isinstance(attribution, str):
            raise ValueError(f"{name}: attribution должен быть строкой.")
        sources.append(
            Source(
                name=name,
                provider=provider,
                source_url=source_url,
                license=license_name.strip(),
                license_verified=verified,
                attribution=attribution.strip(),
                images_are_plate_crops=crops,
                urls=tuple(urls_value),
                dataset=dataset,
                workspace=workspace,
                project=project,
                version=version,
                export_format=export_format,
            )
        )
    return sources


def classify_plate_crop(image: Image.Image) -> tuple[str, str]:
    """Возвращает только эвристическую подсказку для изображения-кропа ГРЗ."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if not width or not height:
        return "unknown", "heuristic_no_dimensions"

    aspect_ratio = width / height
    if 1.15 <= aspect_ratio <= 2.6:
        return "type1a", "aspect_ratio_heuristic"

    hsv_pixels = image.convert("HSV").resize((min(width, 256), min(height, 128))).tobytes()
    pixel_count = len(hsv_pixels) // 3
    yellow = sum(
        1
        for hue, saturation, value in zip(
            hsv_pixels[0::3], hsv_pixels[1::3], hsv_pixels[2::3]
        )
        if 28 <= hue <= 52 and saturation >= 90 and value >= 70
    )
    yellow_fraction = yellow / max(pixel_count, 1)
    if aspect_ratio >= 2.6 and yellow_fraction >= 0.30:
        return "type1b", "HSV_color_heuristic"
    if aspect_ratio >= 2.6:
        return "type1", "aspect_ratio_heuristic"
    return "unknown", "heuristic_uncertain"


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extracts an archive after checking paths, symlinks, and size limits."""
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"Архив превышает лимит {MAX_ARCHIVE_BYTES // (1024 * 1024)} MiB: {archive.name}")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    total_unpacked = 0
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"В архиве слишком много файлов: {archive.name}")
        for member in members:
            name = member.filename.replace("\\", "/")
            pure_name = PurePosixPath(name)
            if pure_name.is_absolute() or ".." in pure_name.parts:
                raise ValueError(f"Небезопасный путь в архиве {archive.name}: {name}")
            mode = member.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise ValueError(f"Символическая ссылка в архиве запрещена: {name}")
            total_unpacked += member.file_size
            if total_unpacked > MAX_UNPACKED_BYTES:
                raise ValueError(f"Распакованный архив превышает лимит: {archive.name}")
            target = (destination / Path(*pure_name.parts)).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Путь выходит за пределы каталога: {name}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def _download_requests(source: Source, destination: Path) -> None:
    """Скачивает изображения только по URL, явно указанным в конфигурации."""
    destination.mkdir(parents=True, exist_ok=True)
    for index, url in enumerate(source.urls, start=1):
        target = destination / f"{index:06d}.download"
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(15, 60),
                allow_redirects=True,
                headers={"User-Agent": "VolgaITOpenDatasetCollector/1.0 (educational project)"},
            ) as response:
                response.raise_for_status()
                length = response.headers.get("Content-Length")
                if length and int(length) > MAX_IMAGE_BYTES:
                    raise ValueError(f"Файл превышает лимит 25 MiB: {url}")
                size = 0
                with target.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > MAX_IMAGE_BYTES:
                            raise ValueError(f"Файл превышает лимит 25 MiB: {url}")
                        output.write(chunk)
        except (requests.RequestException, OSError, ValueError):
            target.unlink(missing_ok=True)
            raise

        try:
            with Image.open(target) as image:
                image_format = image.format
                image.verify()
            suffix = {
                "JPEG": ".jpg",
                "MPO": ".jpg",
                "PNG": ".png",
                "WEBP": ".webp",
                "BMP": ".bmp",
            }.get(image_format or "")
        except (OSError, UnidentifiedImageError) as exc:
            target.unlink(missing_ok=True)
            raise ValueError(f"URL не вернул поддерживаемое изображение: {url}") from exc
        if suffix is None:
            target.unlink(missing_ok=True)
            raise ValueError(f"Неподдерживаемый формат изображения по URL: {url}")
        target.rename(destination / f"{index:06d}{suffix}")


def _download_kaggle(source: Source, destination: Path) -> None:
    """Скачивает Kaggle dataset официальным API и безопасно распаковывает ZIP."""
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError("Для Kaggle задайте переменные окружения KAGGLE_USERNAME и KAGGLE_KEY.")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Для Kaggle установите зависимости из dataset/requirements.txt.") from exc

    destination.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(source.dataset, path=str(destination), unzip=False, quiet=False)
    archives = list(destination.glob("*.zip"))
    if not archives:
        raise RuntimeError(f"Kaggle API не вернул ZIP-архив для {source.dataset}.")
    extracted = destination / "extracted"
    for archive in archives:
        _safe_extract_zip(archive, extracted)


def _download_roboflow(source: Source, destination: Path) -> None:
    """Экспортирует датасет Roboflow через официальный Python SDK."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Для Roboflow задайте переменную окружения ROBOFLOW_API_KEY.")
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError("Для Roboflow установите зависимости из dataset/requirements.txt.") from exc

    project = Roboflow(api_key=api_key).workspace(source.workspace).project(source.project)
    export = project.version(source.version).download(
        source.export_format,
        location=str(destination / "export"),
    )
    location = Path(export.location)
    if not location.exists():
        raise RuntimeError(f"Roboflow SDK сообщил несуществующий каталог: {location}")


def _download_source(source: Source, destination: Path) -> Path:
    """Загружает источник во временный каталог и возвращает каталог изображений."""
    destination.mkdir(parents=True, exist_ok=True)
    if source.provider == "requests":
        _download_requests(source, destination / "downloads")
        return destination / "downloads"
    if source.provider == "kaggle":
        _download_kaggle(source, destination / "kaggle")
        return destination / "kaggle" / "extracted"
    if source.provider == "roboflow":
        _download_roboflow(source, destination / "roboflow")
        return destination / "roboflow" / "export"
    raise ValueError(f"Неподдерживаемый provider: {source.provider}")


def _image_files(root: Path) -> Iterator[Path]:
    """Перечисляет поддерживаемые изображения в каталоге источника."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _ensure_collection_header(path: Path) -> set[str]:
    """Создаёт collection.csv при необходимости и читает уже записанные пути."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8-sig", newline="") as output:
            csv.DictWriter(output, fieldnames=COLLECTION_FIELDS, delimiter=";").writeheader()
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter=";")
        if tuple(reader.fieldnames or ()) != COLLECTION_FIELDS:
            raise ValueError(f"Неожиданный заголовок {path}; файл оставлен без изменений.")
        return {row["image"] for row in reader if row.get("image")}


def collect(
    sources: list[Source],
    dataset_dir: Path,
    include_unverified: bool,
    filter_type: str,
    max_images: int,
) -> tuple[int, int]:
    """Собирает источники, сохраняя provenance отдельно от ручной разметки."""
    collection_path = dataset_dir / "collection.csv"
    existing = _ensure_collection_header(collection_path)
    copied_count = 0
    skipped_count = 0
    for source in sources:
        if not source.license_verified and not include_unverified:
            print(
                f"Пропуск {source.name}: лицензия не подтверждена. "
                "Проверьте права или явно используйте --include-unverified для карантина.",
                file=sys.stderr,
            )
            skipped_count += 1
            continue

        status = "license_verified" if source.license_verified else "quarantine_license_check"
        relative_root = (
            Path("images") / "real" / source.name
            if source.license_verified
            else Path("quarantine") / "unverified" / source.name
        )
        destination_root = dataset_dir / relative_root

        with tempfile.TemporaryDirectory(prefix=f"collect-{source.name}-") as temporary:
            image_root = _download_source(source, Path(temporary))
            for image_path in _image_files(image_root):
                if copied_count >= max_images:
                    print(f"Достигнут лимит --max-images={max_images}.")
                    return copied_count, skipped_count
                try:
                    if image_path.stat().st_size > MAX_IMAGE_BYTES:
                        raise ValueError(f"файл больше {MAX_IMAGE_BYTES // (1024 * 1024)} MiB")
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        if source.images_are_plate_crops:
                            suggestion, method = classify_plate_crop(image)
                        else:
                            suggestion, method = "unknown", "not_a_plate_crop"
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    print(f"Пропуск повреждённого изображения {image_path}: {exc}", file=sys.stderr)
                    skipped_count += 1
                    continue

                if filter_type != "all" and suggestion != filter_type:
                    skipped_count += 1
                    continue

                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                relative_file = relative_root / f"{digest[:16]}{image_path.suffix.lower()}"
                stored_path = dataset_dir / relative_file
                if relative_file.as_posix() in existing:
                    continue
                destination_root.mkdir(parents=True, exist_ok=True)
                if not stored_path.exists():
                    shutil.copyfile(image_path, stored_path)
                row = {
                    "image": relative_file.as_posix(),
                    "candidate_type": suggestion,
                    "suggestion_method": method,
                    "manual_review": "required",
                    "source": source.name,
                    "source_url": source.source_url,
                    "license": source.license,
                    "license_verified": str(source.license_verified).lower(),
                    "attribution": source.attribution,
                    "status": status,
                }
                with collection_path.open("a", encoding="utf-8", newline="") as output:
                    csv.DictWriter(output, fieldnames=COLLECTION_FIELDS, delimiter=";").writerow(row)
                existing.add(relative_file.as_posix())
                copied_count += 1
    return copied_count, skipped_count


def build_parser() -> argparse.ArgumentParser:
    """Строит интерфейс командной строки."""
    parser = argparse.ArgumentParser(
        description="Сбор открытых изображений с записью происхождения и статуса лицензии."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DATASET_DIR / "sources.json",
        help="JSON-конфигурация; по умолчанию dataset/sources.json.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Каталог датасета (по умолчанию каталог этого скрипта).",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Разрешить скачивание непроверенных лицензий только в quarantine/unverified/.",
    )
    parser.add_argument(
        "--filter-type",
        choices=("all", "type1", "type1a", "type1b", "unknown"),
        default="all",
        help="Оставить только указанную эвристическую подсказку (только для crop-изображений).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Максимум новых изображений за запуск (по умолчанию 1000).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_images < 1:
        parser.error("--max-images должен быть больше нуля.")
    try:
        sources = load_sources(args.config)
        copied, skipped = collect(
            sources,
            args.dataset_dir,
            args.include_unverified,
            args.filter_type,
            args.max_images,
        )
    except (ValueError, RuntimeError, requests.RequestException, zipfile.BadZipFile) as exc:
        print(f"Ошибка сбора: {exc}", file=sys.stderr)
        return 1
    print(f"Добавлено изображений: {copied}; пропущено: {skipped}.")
    print("Эвристические подсказки не являются проверенной разметкой; проверьте каждый файл вручную.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
