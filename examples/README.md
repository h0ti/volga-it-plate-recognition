# Пример изображения

`4373_russian_plate_cc0.jpg` — фотография с российским номером, Wikimedia Commons,
лицензия **CC0** (автор Rakoon; атрибуция не требуется, но указана).

Источник: <https://commons.wikimedia.org/wiki/File:4373_Russian_license_plate_Toyota_car.jpg>

Файл добавлен только для быстрой проверки запуска:

```powershell
python solution\predict.py --input examples ^
    --weights solution\weights\best_v2.pt ^
    --ocr-weights solution\weights\ocr_v3.pt ^
    --output predictions.csv --device 0
```

Эталонный номер на фото — `C051KO199` (`type1`), он входит в журнал ручной разметки
`dataset/real_review_inbox/review.csv`. Текущая модель читает его как `C051KO1#`:
одна позиция кода региона не распознана и честно заменена на `#`. Это ожидаемое
поведение MVP, а не гарантия точности на произвольных кадрах.
