#!/usr/bin/env python3
"""
Скрипт обучения YOLO11 модели для детекции switch на электрических схемах.
"""

from ultralytics import YOLO

def main():
    print("=" * 60)
    print("Начинаем обучение модели YOLO11n для детекции switch")
    print("=" * 60)

    # Загружаем предобученную модель YOLO11n
    model = YOLO("yolo11n.pt")

    # Параметры обучения
    results = model.train(
        data="data.yaml",
        epochs=150,
        imgsz=1280,
        batch=2,

        # Отключаем геометрические аугментации
        mosaic=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        fliplr=0.0,
        flipud=0.0,

        # Цветовые аугментации (умеренные)
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,

        # Другие параметры
        patience=50,
        save=True,
        save_period=10,
        device="cpu",  # Используем CPU (можно заменить на "0" для GPU)
        workers=2,
        project="runs/detect",
        name="train",
        exist_ok=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Обучение завершено!")
    print(f"Лучшая модель сохранена в: runs/detect/train/weights/best.pt")
    print(f"Последняя модель сохранена в: runs/detect/train/weights/last.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
