#!/usr/bin/env python3
"""
Скрипт для визуализации предсказаний обученной модели с настраиваемым порогом confidence.
Создаёт side-by-side сравнение: predictions vs ground truth.

Использование:
    python predict_and_visualize.py --model runs/detect/quick_train/weights/best.pt --image data/images/train/train_00.jpg --conf 0.25
    python predict_and_visualize.py --model runs/detect/quick_train/weights/best.pt --image data/images/train/train_00.jpg --conf 0.35 --output custom_result.jpg
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import cv2
import numpy as np


def predict_and_visualize(model_path, image_path, conf_threshold=0.25, output_path=None):
    """
    Делает предсказание и создаёт side-by-side визуализацию.

    Args:
        model_path: путь к обученной модели (.pt файл)
        image_path: путь к изображению для предсказания
        conf_threshold: порог confidence (0.0-1.0)
        output_path: путь для сохранения результата (опционально)
    """

    # Конвертируем пути в Path объекты
    model_path = Path(model_path)
    image_path = Path(image_path)

    # Проверяем существование файлов
    if not model_path.exists():
        print(f"❌ Модель не найдена: {model_path}")
        return

    if not image_path.exists():
        print(f"❌ Изображение не найдено: {image_path}")
        return

    print("=" * 70)
    print("🔍 ВИЗУАЛИЗАЦИЯ ПРЕДСКАЗАНИЙ")
    print("=" * 70)
    print(f"\n📦 Модель: {model_path}")
    print(f"📷 Изображение: {image_path}")
    print(f"🎯 Порог confidence: {conf_threshold}")
    print(f"📐 Размер изображения: {image_path.stat().st_size / 1024:.1f} KB\n")

    # Загружаем модель
    print("⏳ Загрузка модели...")
    model = YOLO(str(model_path))
    print("✓ Модель загружена\n")

    # Делаем предсказание
    print("⏳ Выполнение предсказания...")
    results = model.predict(
        source=str(image_path),
        imgsz=1280,
        conf=conf_threshold,
        iou=0.45,
        save=False,
        verbose=False,
    )

    # Получаем результаты
    result = results[0]
    boxes = result.boxes

    print(f"✓ Предсказание выполнено: найдено {len(boxes)} объектов\n")

    if len(boxes) == 0:
        print("❌ Объекты не найдены!")
        print("💡 Попробуйте:")
        print(f"   - Снизить conf_threshold (текущий: {conf_threshold})")
        print("   - Использовать другую модель")
        print("   - Проверить изображение\n")
        return

    # Читаем изображение
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Не удалось прочитать изображение: {image_path}")
        return

    # Создаём копии для рисования
    prediction_img = img.copy()
    gt_img = img.copy()

    # Получаем информацию о детекциях
    confidences = boxes.conf.cpu().numpy()
    boxes_xyxy = boxes.xyxy.cpu().numpy()

    # ===== ЛЕВАЯ ЧАСТЬ: PREDICTIONS =====
    for i, (box, conf) in enumerate(zip(boxes_xyxy, confidences)):
        conf_val = float(conf)
        x1, y1, x2, y2 = map(int, box)

        # Цвет по уровню confidence
        if conf_val >= 0.7:
            color = (0, 255, 0)      # Зелёный = отлично
        elif conf_val >= 0.5:
            color = (0, 200, 255)    # Оранжевый = хорошо
        elif conf_val >= 0.3:
            color = (0, 150, 255)    # Светло-оранжевый = средне
        else:
            color = (0, 100, 255)    # Красный = слабо

        # Рисуем рамку
        cv2.rectangle(prediction_img, (x1, y1), (x2, y2), color, 3)

        # Подпись
        label = f"switch {conf_val:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Фон для текста
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(
            prediction_img,
            (x1, y1 - text_height - baseline - 8),
            (x1 + text_width + 5, y1),
            color,
            -1
        )

        # Текст
        cv2.putText(
            prediction_img,
            label,
            (x1 + 2, y1 - baseline - 5),
            font,
            font_scale,
            (0, 0, 0),  # Чёрный текст на цветном фоне
            thickness
        )

    # Добавляем информационную панель вверху (PREDICTIONS)
    info_text = f"PREDICTIONS: {len(boxes)} detections (conf >= {conf_threshold:.2f})"
    cv2.rectangle(prediction_img, (0, 0), (img.shape[1], 50), (255, 255, 255), -1)
    cv2.putText(
        prediction_img,
        info_text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2
    )

    # ===== ПРАВАЯ ЧАСТЬ: GROUND TRUTH =====
    # Ищем файл разметки (два возможных пути)
    # Вариант 1: data/labels/train/image.txt (стандартная структура YOLO)
    label_path = Path("data/labels") / image_path.parent.name / (image_path.stem + ".txt")

    # Вариант 2: если не найдена, ищем рядом с images
    if not label_path.exists():
        label_path = image_path.parent.parent / "labels" / image_path.parent.name / (image_path.stem + ".txt")

    gt_boxes = []

    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class x_center y_center width height (normalized 0-1)
                    cls = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Конвертируем в пиксельные координаты
                    img_h, img_w = img.shape[:2]
                    x_center_px = x_center * img_w
                    y_center_px = y_center * img_h
                    width_px = width * img_w
                    height_px = height * img_h

                    # Вычисляем x1, y1, x2, y2
                    x1 = int(x_center_px - width_px / 2)
                    y1 = int(y_center_px - height_px / 2)
                    x2 = int(x_center_px + width_px / 2)
                    y2 = int(y_center_px + height_px / 2)

                    gt_boxes.append((x1, y1, x2, y2, cls))

        # Рисуем ground truth boxes
        gt_color = (0, 255, 0)  # Зелёный для GT
        for i, (x1, y1, x2, y2, cls) in enumerate(gt_boxes):
            # Рисуем рамку
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), gt_color, 3)

            # Подпись
            label = f"GT switch #{i+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Фон для текста
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(
                gt_img,
                (x1, y1 - text_height - baseline - 8),
                (x1 + text_width + 5, y1),
                gt_color,
                -1
            )

            # Текст
            cv2.putText(
                gt_img,
                label,
                (x1 + 2, y1 - baseline - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )

        # Добавляем информационную панель вверху (GROUND TRUTH)
        gt_info_text = f"GROUND TRUTH: {len(gt_boxes)} objects"
        cv2.rectangle(gt_img, (0, 0), (img.shape[1], 50), (255, 255, 255), -1)
        cv2.putText(
            gt_img,
            gt_info_text,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2
        )
    else:
        # Если разметки нет, показываем только predictions
        print(f"⚠️  Ground truth разметка не найдена: {label_path}")
        print("   Будет показано только изображение с предсказаниями\n")
        gt_img = None

    # ===== СКЛЕИВАЕМ ДВА ИЗОБРАЖЕНИЯ =====
    if gt_img is not None:
        # Добавляем разделитель между изображениями (белая полоса)
        separator = np.ones((img.shape[0], 10, 3), dtype=np.uint8) * 255
        combined_img = np.hstack([prediction_img, separator, gt_img])
    else:
        # Только predictions
        combined_img = prediction_img

    # Определяем путь для сохранения
    if output_path is None:
        # По умолчанию сохраняем в results/
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / f"prediction_{image_path.stem}_conf{conf_threshold:.2f}.jpg"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Сохраняем
    cv2.imwrite(str(output_path), combined_img)

    print("=" * 70)
    print("✅ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)
    print(f"💾 Результат сохранён: {output_path}")
    print(f"   Размер: {output_path.stat().st_size / 1024:.1f} KB")

    if gt_img is not None:
        print(f"\n   Левая часть: Predictions ({len(boxes)} детекций)")
        print(f"   Правая часть: Ground Truth ({len(gt_boxes)} объектов)")
    else:
        print(f"\n   Predictions: {len(boxes)} детекций")

    # Статистика по confidence
    print("\n" + "=" * 70)
    print("📊 СТАТИСТИКА ПО УВЕРЕННОСТИ")
    print("=" * 70)
    print(f"Минимальная confidence: {min(confidences):.3f}")
    print(f"Максимальная confidence: {max(confidences):.3f}")
    print(f"Средняя confidence:     {sum(confidences)/len(confidences):.3f}")

    # Распределение по диапазонам
    print("\n📊 Распределение по диапазонам:")
    ranges = [
        (0.9, 1.0, "0.9-1.0 (отлично)"),
        (0.7, 0.9, "0.7-0.9 (хорошо)"),
        (0.5, 0.7, "0.5-0.7 (средне)"),
        (0.3, 0.5, "0.3-0.5 (слабо)"),
        (0.0, 0.3, "0.0-0.3 (очень слабо)"),
    ]

    for min_conf, max_conf, label in ranges:
        count = sum(1 for c in confidences if min_conf <= c < max_conf)
        if count > 0:
            percentage = (count / len(confidences)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {label:<25} {count:>3} шт. ({percentage:>5.1f}%) {bar}")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Визуализация предсказаний обученной YOLO модели с настраиваемым порогом confidence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Базовое использование (conf=0.25 по умолчанию)
  python predict_and_visualize.py --model runs/detect/quick_train/weights/best.pt --image data/images/train/train_00.jpg

  # С настраиваемым порогом confidence
  python predict_and_visualize.py --model runs/detect/quick_train/weights/best.pt --image data/images/train/train_00.jpg --conf 0.35

  # С указанием пути для сохранения результата
  python predict_and_visualize.py --model runs/detect/quick_train/weights/best.pt --image data/images/train/train_00.jpg --conf 0.40 --output my_result.jpg

  # Тестирование на test изображениях
  python predict_and_visualize.py --model runs/detect/quick_train/weights/best.pt --image data/images/test/test_01.jpg --conf 0.25
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="runs/detect/quick_train/weights/best.pt",
        help="Путь к обученной модели (.pt файл). По умолчанию: runs/detect/quick_train/weights/best.pt"
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Путь к изображению для предсказания (обязательный параметр)"
    )

    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Порог confidence (0.0-1.0). По умолчанию: 0.25"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Путь для сохранения результата. По умолчанию: results/prediction_<имя>_conf<порог>.jpg"
    )

    args = parser.parse_args()

    # Валидация confidence threshold
    if not 0.0 <= args.conf <= 1.0:
        print("❌ Ошибка: confidence threshold должен быть в диапазоне 0.0-1.0")
        sys.exit(1)

    # Запускаем визуализацию
    predict_and_visualize(
        model_path=args.model,
        image_path=args.image,
        conf_threshold=args.conf,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
