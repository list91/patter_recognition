#!/usr/bin/env python3
"""
Скрипт для обучения модели и быстрой проверки на тренировочном изображении
с группировкой по confidence.
"""

import yaml
import sys
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import cv2
import numpy as np
import shutil
import random

def load_config(config_path="train_config.yaml"):
    """Загружает конфигурацию из YAML файла."""
    if not Path(config_path).exists():
        print(f"⚠️  Конфиг {config_path} не найден, используем параметры по умолчанию")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def convert_to_grayscale_dataset(source_dir, target_dir):
    """Конвертирует изображения в grayscale и копирует в целевую директорию."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Создаем целевую директорию
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"🎨 Конвертация изображений в черно-белые...")
    print(f"   Источник: {source_path}")
    print(f"   Цель: {target_path}")

    converted_count = 0
    for img_file in source_path.glob("*.jpg"):
        # Читаем изображение
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        # Конвертируем в grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Конвертируем обратно в 3-канальное (дублируем канал)
        # Это нужно потому что YOLO ожидает 3-канальное изображение
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Сохраняем
        target_file = target_path / img_file.name
        cv2.imwrite(str(target_file), gray_3ch)
        converted_count += 1

    print(f"   ✓ Конвертировано: {converted_count} изображений\n")
    return target_path

def prepare_grayscale_dataset():
    """Подготавливает черно-белый датасет для обучения."""
    # Конвертируем train изображения
    train_source = Path("data/images/train")
    train_target = Path("data/images/train_grayscale")
    convert_to_grayscale_dataset(train_source, train_target)

    # Копируем labels
    labels_source = Path("data/labels/train")
    labels_target = Path("data/labels/train_grayscale")

    labels_target.mkdir(parents=True, exist_ok=True)

    for label_file in labels_source.glob("*.txt"):
        shutil.copy2(label_file, labels_target / label_file.name)

    print(f"   ✓ Скопированы метки в {labels_target}\n")

    # Создаем временный data.yaml для grayscale
    data_yaml_content = """path: ./data
train: images/train_grayscale
val: images/train_grayscale
nc: 1
names: ['switch']
"""

    with open("data_grayscale.yaml", 'w') as f:
        f.write(data_yaml_content)

    print(f"   ✓ Создан data_grayscale.yaml\n")

    return "data_grayscale.yaml"

def train_model(config):
    """Обучает модель по конфигурации."""
    print("=" * 70)
    print("🚀 ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 70)
    print(f"\n📋 Эксперимент: {config.get('experiment_name', 'default')}")
    print(f"📦 Модель: {config.get('model', 'yolo11n.pt')}")
    print(f"⏱️  Эпохи: {config.get('epochs', 150)}")
    print(f"📐 Размер: {config.get('imgsz', 1280)}")
    print(f"🎯 Confidence threshold: {config.get('conf_threshold', 0.1)}")

    # Проверяем режим grayscale
    use_grayscale = config.get('grayscale', False)
    print(f"🎨 Режим: {'Черно-белый (grayscale)' if use_grayscale else 'Цветной (RGB)'}\n")

    # Подготавливаем датасет
    if use_grayscale:
        data_yaml = prepare_grayscale_dataset()
    else:
        data_yaml = "data.yaml"

    # Загружаем модель
    model = YOLO(config.get('model', 'yolo11n.pt'))

    # Запускаем обучение
    print("⏳ Обучение началось...\n")
    print("⚠️  ВНИМАНИЕ: Используются встроенные дефолты YOLO (гиперпараметры НЕ передаются)\n")

    results = model.train(
        data=data_yaml,

        # Только технические параметры (НЕ гиперпараметры обучения)
        device="cpu",
        workers=2,
        project="runs/detect",
        name="quick_train",
        exist_ok=True,  # Перезаписываем
        verbose=False,  # Меньше вывода
        plots=True,
    )

    # Получаем путь к модели
    best_model_path = Path(model.trainer.save_dir) / "weights" / "best.pt"

    print("\n" + "=" * 70)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"📁 Модель сохранена: {best_model_path}\n")

    return best_model_path

def test_on_train_image(model_path, conf_threshold=0.1, use_grayscale=False):
    """Тестирует модель на тренировочном изображении."""
    print("=" * 70)
    print("🔍 ТЕСТИРОВАНИЕ НА ТРЕНИРОВОЧНОМ ИЗОБРАЖЕНИИ")
    print("=" * 70)

    train_image = Path("data/images/train/train_00.jpg")

    if not train_image.exists():
        print(f"❌ Тренировочное изображение не найдено: {train_image}")
        return

    print(f"\n📷 Изображение: {train_image}")
    print(f"🎯 Порог confidence: {conf_threshold}")
    print(f"🎨 Режим: {'Черно-белый (grayscale)' if use_grayscale else 'Цветной (RGB)'}\n")

    # Подготавливаем изображение
    if use_grayscale:
        # Читаем и конвертируем в grayscale
        img = cv2.imread(str(train_image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Сохраняем временно
        temp_image = Path("data/images/train/train_00_grayscale_temp.jpg")
        cv2.imwrite(str(temp_image), gray_3ch)
        test_image = temp_image
    else:
        test_image = train_image

    # Загружаем модель
    model = YOLO(model_path)

    # Делаем предсказание
    results = model.predict(
        source=str(test_image),
        imgsz=1280,
        conf=conf_threshold,
        iou=0.45,
        save=False,
        verbose=False,
    )

    # Получаем результаты
    result = results[0]
    boxes = result.boxes

    if len(boxes) == 0:
        print("❌ Объекты не найдены!")
        print("💡 Попробуйте:")
        print("   - Снизить conf_threshold")
        print("   - Увеличить epochs")
        print("   - Использовать большую модель\n")

        # Удаляем временный файл
        if use_grayscale and temp_image.exists():
            temp_image.unlink()
        return

    confidences = boxes.conf.cpu().numpy()

    # Читаем оригинальное изображение для рисования
    if use_grayscale:
        img = cv2.imread(str(test_image))
    else:
        img = cv2.imread(str(train_image))

    # Создаем PREDICTION копию для рисования
    prediction_img = img.copy()

    # Создаем GROUND TRUTH копию для рисования
    gt_img = img.copy()

    # Округляем до сотых и считаем
    rounded_confs = [round(float(c), 2) for c in confidences]
    conf_counts = Counter(rounded_confs)

    # Получаем координаты boxes
    boxes_xyxy = boxes.xyxy.cpu().numpy()

    # ===== ЛЕВАЯ ЧАСТЬ: PREDICTIONS =====
    for i, (box, conf) in enumerate(zip(boxes_xyxy, confidences)):
        conf_val = float(conf)
        x1, y1, x2, y2 = map(int, box)

        # Цвет по уровню confidence
        if conf_val >= 0.4:
            color = (0, 255, 0)      # Зелёный = высокая
            level = "HIGH"
        elif conf_val >= 0.25:
            color = (0, 200, 255)    # Оранжевый = средняя
            level = "MEDIUM"
        else:
            color = (0, 100, 255)    # Красный = низкая
            level = "LOW"

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
    # Читаем ground truth labels
    label_path = Path("data/labels/train") / (train_image.stem + ".txt")
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

    # ===== СКЛЕИВАЕМ ДВА ИЗОБРАЖЕНИЯ =====
    # Добавляем разделитель между изображениями (белая полоса)
    separator = np.ones((img.shape[0], 10, 3), dtype=np.uint8) * 255
    combined_img = np.hstack([prediction_img, separator, gt_img])

    # Создаем папку results если её нет
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Сохраняем (перезаписываем предыдущий результат)
    output_path = results_dir / "train_prediction.jpg"
    cv2.imwrite(str(output_path), combined_img)
    print(f"💾 Визуализация сохранена: {output_path}\n")
    print(f"   Левая часть: Predictions ({len(boxes)} детекций)")
    print(f"   Правая часть: Ground Truth ({len(gt_boxes)} объектов)\n")

    # Удаляем временный файл
    if use_grayscale and temp_image.exists():
        temp_image.unlink()

    # Сортируем по confidence (от большего к меньшему)
    sorted_confs = sorted(conf_counts.items(), reverse=True)

    # Выводим результаты
    print("=" * 70)
    print("📊 СТАТИСТИКА ПО УВЕРЕННОСТИ")
    print("=" * 70)
    print(f"\n{'Confidence':<15} {'Количество объектов':>20}")
    print("-" * 70)

    total = 0
    for conf, count in sorted_confs:
        print(f"{conf:<15.2f} {count:>20} шт.")
        total += count

    print("-" * 70)
    print(f"{'ВСЕГО:':<15} {total:>20} шт.")

    # Дополнительная статистика
    print("\n" + "=" * 70)
    print("📈 ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА")
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
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {label:<25} {count:>3} шт. ({percentage:>5.1f}%) {bar}")

    print("\n" + "=" * 70 + "\n")

def main():
    # Загружаем конфигурацию
    config_file = sys.argv[1] if len(sys.argv) > 1 else "train_config.yaml"

    print("\n" + "=" * 70)
    print("🎯 БЫСТРЫЙ WORKFLOW: ОБУЧЕНИЕ → ТЕСТ → СТАТИСТИКА")
    print("=" * 70)
    print(f"📄 Конфигурация: {config_file}\n")

    config = load_config(config_file)

    # Параметры по умолчанию
    # defaults = {
    #     'experiment_name': 'quick_test',
    #     'model': 'yolo11n.pt',
    #     'epochs': 150,
    #     'imgsz': 1280,
    #     'batch': 2,
    #     'optimizer': 'auto',
    #     'lr0': 0.002,
    #     'lrf': 0.01,
    #     'momentum': 0.937,
    #     'weight_decay': 0.0005,
    #     'mosaic': 0.0,
    #     'degrees': 0.0,
    #     'translate': 0.0,
    #     'scale': 0.0,
    #     'fliplr': 0.0,
    #     'flipud': 0.0,
    #     'shear': 0.0,
    #     'hsv_h': 0.015,
    #     'hsv_s': 0.5,
    #     'hsv_v': 0.4,
    #     'dropout': 0.0,
    #     'mixup': 0.0,
    #     'copy_paste': 0.0,
    #     'box': 7.5,
    #     'cls': 0.5,
    #     'dfl': 1.5,
    #     'patience': 50,
    #     'close_mosaic': 10,
    #     'warmup_epochs': 3.0,
    #     'cos_lr': False,
    #     'conf_threshold': 0.1,
    #     'iou_threshold': 0.45,
    #     'grayscale': False,
    # }

    # Объединяем с дефолтами
    # for key, value in defaults.items():
    #     config.setdefault(key, value)

    # 1. Обучение
    model_path = train_model(config)

    # 2. Тестирование на тренировочном изображении
    conf_threshold = config.get('conf_threshold', 0.1)
    use_grayscale = config.get('grayscale', False)
    test_on_train_image(model_path, conf_threshold, use_grayscale)

    print("✅ Готово! Измените параметры в train_config.yaml и запустите снова\n")

if __name__ == "__main__":
    main()
