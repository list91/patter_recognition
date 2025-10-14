#!/usr/bin/env python3
"""
Скрипт для тестирования обученной модели на неразмеченных схемах
и визуализации результатов детекции.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2

def main():
    print("=" * 60)
    print("Тестирование обученной модели и визуализация результатов")
    print("=" * 60)

    # Путь к лучшей модели
    model_path = "runs/detect/train/weights/best.pt"

    # Проверяем наличие модели
    if not os.path.exists(model_path):
        print(f"ОШИБКА: Модель не найдена по пути {model_path}")
        print("Сначала запустите train.py для обучения модели!")
        return

    # Загружаем обученную модель
    print(f"\nЗагружаем модель из {model_path}...")
    model = YOLO(model_path)

    # Папка с тестовыми изображениями
    test_images_dir = Path("data/images/test")

    # Проверяем наличие тестовых изображений
    if not test_images_dir.exists():
        print(f"ОШИБКА: Папка {test_images_dir} не найдена!")
        return

    # Получаем список всех изображений
    test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))

    if not test_images:
        print(f"ОШИБКА: Нет изображений в папке {test_images_dir}!")
        return

    print(f"Найдено {len(test_images)} тестовых изображений\n")

    # Создаём папку для результатов
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Обрабатываем каждое изображение
    total_detections = 0
    for img_path in test_images:
        print(f"Обработка: {img_path.name}...")

        # Делаем предсказание
        results = model.predict(
            source=str(img_path),
            imgsz=1280,
            conf=0.1,  # Порог уверенности (снижен для лучшей детекции)
            iou=0.45,
            save=False,
            verbose=False,
        )

        # Получаем результаты
        result = results[0]
        num_detections = len(result.boxes)
        total_detections += num_detections

        print(f"  Найдено switch: {num_detections}")

        # Сохраняем изображение с предсказаниями
        output_path = results_dir / f"pred_{img_path.name}"

        # Рисуем bounding boxes на изображении
        annotated_img = result.plot(
            line_width=2,
            font_size=12,
            conf=True,
            labels=True,
        )

        # Сохраняем результат
        cv2.imwrite(str(output_path), annotated_img)
        print(f"  Результат сохранён: {output_path}\n")

    print("=" * 60)
    print("Тестирование завершено!")
    print(f"Всего найдено объектов switch: {total_detections}")
    print(f"Результаты сохранены в папке: {results_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
