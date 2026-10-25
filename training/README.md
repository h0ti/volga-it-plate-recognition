# Обучение и оценка

Все скрипты офлайновые, фиксируют seed и не скачивают данные. Требуется Python 3.10+
и CUDA для обучения; инференс работает и на CPU.

## Порядок

```powershell
# окружение (один раз)
python -m venv training\.venv
training\.venv\Scripts\python.exe -m pip install -r training\requirements.txt

# 1. синтетический датасет и детектор
training\.venv\Scripts\python.exe dataset\generator\build_dataset.py --count 6000 --out dataset\generated_4060
training\.venv\Scripts\python.exe training\train_detector.py --dataset dataset\generated_4060 --out training\run_plates_4060 --epochs 40

# 2. датасет дообучения на реальных кадрах и fine-tune детектора
training\.venv\Scripts\python.exe training\build_det_v2_dataset.py
training\.venv\Scripts\python.exe training\train_det_v2.py --epochs 14

# 3. OCR на реальных кропах + сценах
training\.venv\Scripts\python.exe training\train_ocr_v2.py --epochs 14

# 4. оценка на ручной разметке
training\.venv\Scripts\python.exe training\eval_real_pipeline.py ^
    --weights solution\weights\best_v2.pt --ocr-weights solution\weights\ocr_v2.pt
```

## Что делает каждый скрипт

| Скрипт | Назначение |
|---|---|
| `train_detector.py` | конвертирует синтетическую разметку в YOLO-формат и обучает YOLOv8n |
| `build_det_v2_dataset.py` | собирает mixed-датасет: синтетика + реальные кадры (bbox `x1,y1,x2,y2` из журнала) + негативы |
| `train_det_v2.py` | fine-tune детектора на mixed-датасете (AdamW, mosaic, без fliplr) |
| `train_ocr.py` | первая версия OCR только на чистых рендерах (baseline) |
| `train_ocr_v2.py` | OCR на реальных кропах + синтетических сценах + рендерах, с фото-аугментациями |
| `eval_real_pipeline.py` | детекция/тип/OCR/сквозная метрика на ручной разметке |

## Особенности окружения

- Ultralytics сканирует метки через process pool; в ограниченных Windows-песочницах
  именованные каналы запрещены, поэтому `train_det_v2.py` подменяет пул на
  последовательный (`_SerialPool`). На обычной машине это не требуется.
- `training/run_*` — локальные артефакты обучения, в репозиторий не входят
  (см. `.gitignore`). Финальные веса лежат в `solution/weights/`.

## Ограничения

- Синтетическая валидация (mAP50 0.97) не равна качеству на реальных камерах.
- 184 ручные аннотации — мало; тип 1А почти не представлен.
- Для повторного обучения OCR нужны локальные копии `data/commons/images`
  (в репозиторий не входят по лицензионным причинам).
