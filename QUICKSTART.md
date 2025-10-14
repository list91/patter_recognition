# Быстрый старт

## 1. Установите зависимости

```bash
pip install -r requirements.txt
```

## 2. Запустите обучение и тестирование

### Windows
```bash
run_all.bat
```

### Linux/macOS
```bash
./run_all.sh
```

## 3. Проверьте результаты

После завершения работы скриптов вы найдёте:

**Обученную модель:**
- `runs/detect/train/weights/best.pt`

**Графики обучения:**
- `runs/detect/train/results.png`
- `runs/detect/train/confusion_matrix.png`

**Результаты детекции:**
- `results/pred_test_01.jpg`
- `results/pred_test_02.jpg`
- `results/pred_test_03.jpg`
- `results/pred_test_04.jpg`

## Что делает каждый скрипт?

### train.py
- Загружает предобученную модель YOLO11n
- Обучает её на размеченных данных (1 изображение)
- Сохраняет веса и метрики обучения

### test_and_visualize.py
- Загружает обученную модель
- Делает предсказания на тестовых изображениях
- Рисует bounding boxes вокруг найденных объектов
- Сохраняет результаты в папку `results/`

## Настройка параметров

Если хотите изменить параметры обучения, отредактируйте `train.py`:

```python
results = model.train(
    epochs=150,      # Количество эпох
    imgsz=1280,      # Размер изображения
    batch=2,         # Размер батча
    # другие параметры...
)
```

Для использования GPU замените:
```python
device="cpu"  # на
device="0"    # для первой GPU
```

## Добавление данных

1. Поместите новые изображения в `data/images/train/`
2. Создайте соответствующие `.txt` файлы в `data/labels/train/`
3. Запустите `train.py` снова

Формат аннотации (YOLO):
```
0 center_x center_y width height
```
Все координаты нормализованы в диапазоне [0, 1].
