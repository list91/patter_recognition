#!/usr/bin/env python3
"""
Скрипт для тестирования обученной модели на неразмеченных схемах
с детальной статистикой и визуализацией.
"""

import os
import sys
import json
from pathlib import Path
from ultralytics import YOLO
import cv2

def load_experiment_config(run_dir):
    """Загружает конфигурацию эксперимента."""
    config_path = Path(run_dir) / 'experiment_info.json'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def main():
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ И ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 70)

    # Определяем путь к модели
    if len(sys.argv) > 1:
        # Если передано имя эксперимента
        exp_name = sys.argv[1]
        model_path = f"runs/detect/{exp_name}/weights/best.pt"
        run_dir = f"runs/detect/{exp_name}"
    else:
        # Используем train (первый запуск)
        model_path = "runs/detect/train/weights/best.pt"
        run_dir = "runs/detect/train"

    # Проверяем наличие модели
    if not os.path.exists(model_path):
        print(f"❌ ОШИБКА: Модель не найдена по пути {model_path}")
        print("Сначала запустите train.py для обучения модели!")
        return

    print(f"\n📂 Используется модель: {model_path}")

    # Загружаем конфигурацию эксперимента
    exp_config = load_experiment_config(run_dir)
    if exp_config:
        conf_threshold = exp_config['config'].get('conf_threshold', 0.1)
        iou_threshold = exp_config['config'].get('iou_threshold', 0.45)
        print(f"⚙️  Конфигурация из эксперимента:")
        print(f"   - conf_threshold: {conf_threshold}")
        print(f"   - iou_threshold: {iou_threshold}")
    else:
        # Значения по умолчанию
        conf_threshold = 0.1
        iou_threshold = 0.45
        print(f"⚙️  Используются параметры по умолчанию")

    # Загружаем обученную модель
    print(f"\n⏳ Загружаем модель...")
    model = YOLO(model_path)

    # Папка с тестовыми изображениями
    test_images_dir = Path("data/images/test")

    # Проверяем наличие тестовых изображений
    if not test_images_dir.exists():
        print(f"❌ ОШИБКА: Папка {test_images_dir} не найдена!")
        return

    # Получаем список всех изображений
    test_images = sorted(list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png")))

    if not test_images:
        print(f"❌ ОШИБКА: Нет изображений в папке {test_images_dir}!")
        return

    print(f"✅ Найдено {len(test_images)} тестовых изображений")
    print(f"📊 Порог уверенности: {conf_threshold}")
    print(f"📊 Порог IoU: {iou_threshold}\n")

    # Создаём папку для результатов
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Статистика
    total_detections = 0
    stats_per_image = []

    # Обрабатываем каждое изображение
    print("=" * 70)
    print("🔍 ОБРАБОТКА ИЗОБРАЖЕНИЙ")
    print("=" * 70 + "\n")

    for img_path in test_images:
        print(f"📷 {img_path.name}")

        # Делаем предсказание
        results = model.predict(
            source=str(img_path),
            imgsz=1280,
            conf=conf_threshold,
            iou=iou_threshold,
            save=False,
            verbose=False,
        )

        # Получаем результаты
        result = results[0]
        boxes = result.boxes
        num_detections = len(boxes)
        total_detections += num_detections

        # Собираем статистику по confidence
        confidences = boxes.conf.cpu().numpy() if num_detections > 0 else []

        img_stats = {
            'name': img_path.name,
            'detections': num_detections,
            'avg_conf': float(confidences.mean()) if len(confidences) > 0 else 0.0,
            'max_conf': float(confidences.max()) if len(confidences) > 0 else 0.0,
            'min_conf': float(confidences.min()) if len(confidences) > 0 else 0.0,
        }
        stats_per_image.append(img_stats)

        # Выводим детали
        if num_detections > 0:
            print(f"   ✓ Найдено: {num_detections} switch")
            print(f"   ├─ Avg confidence: {img_stats['avg_conf']:.3f}")
            print(f"   ├─ Max confidence: {img_stats['max_conf']:.3f}")
            print(f"   └─ Min confidence: {img_stats['min_conf']:.3f}")
        else:
            print(f"   ✗ Объекты не найдены")

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
        print(f"   💾 Сохранено: {output_path}\n")

    # Итоговая статистика
    print("=" * 70)
    print("📈 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 70 + "\n")

    print(f"🎯 Всего изображений: {len(test_images)}")
    print(f"🔍 Всего найдено объектов: {total_detections}")
    print(f"📊 Среднее на изображение: {total_detections / len(test_images):.1f}")

    # Группировка по категориям
    found = [s for s in stats_per_image if s['detections'] > 0]
    not_found = [s for s in stats_per_image if s['detections'] == 0]

    print(f"\n📋 С детекциями: {len(found)}/{len(test_images)}")
    print(f"📋 Без детекций: {len(not_found)}/{len(test_images)}")

    if found:
        avg_conf_all = sum(s['avg_conf'] for s in found) / len(found)
        print(f"\n💯 Средняя уверенность: {avg_conf_all:.3f}")

    # Топ-5 по количеству детекций
    print("\n🏆 ТОП-5 ПО КОЛИЧЕСТВУ ДЕТЕКЦИЙ:")
    top5 = sorted(stats_per_image, key=lambda x: x['detections'], reverse=True)[:5]
    for i, s in enumerate(top5, 1):
        print(f"   {i}. {s['name']:20} - {s['detections']:3} объектов (conf: {s['avg_conf']:.3f})")

    # Проверка тренировочного изображения
    train_image_stats = next((s for s in stats_per_image if 'train_01' in s['name']), None)
    if train_image_stats:
        print("\n📌 ПРОВЕРКА НА ТРЕНИРОВОЧНОМ ИЗОБРАЖЕНИИ:")
        print(f"   Изображение: {train_image_stats['name']}")
        print(f"   Найдено: {train_image_stats['detections']} switch")
        print(f"   Avg confidence: {train_image_stats['avg_conf']:.3f}")
        if train_image_stats['detections'] == 0:
            print("   ⚠️  Модель не нашла объекты даже на тренировочном изображении!")
            print("   💡 Попробуйте снизить conf_threshold или обучить дольше")
        elif train_image_stats['avg_conf'] < 0.5:
            print("   ⚠️  Низкая уверенность на тренировочном изображении")
            print("   💡 Модель переобучается плохо - нужно больше эпох или данных")
        else:
            print("   ✅ Модель хорошо работает на тренировочном изображении")

    # Сохраняем статистику в JSON
    stats_path = results_dir / "test_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_path': model_path,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'total_images': len(test_images),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(test_images),
            'images_with_detections': len(found),
            'images_without_detections': len(not_found),
            'per_image_stats': stats_per_image,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Статистика сохранена: {stats_path}")

    print("\n" + "=" * 70)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"📁 Результаты сохранены в: {results_dir}")
    print(f"📊 Статистика: {stats_path}")

    # Рекомендации
    print("\n" + "=" * 70)
    print("💡 РЕКОМЕНДАЦИИ")
    print("=" * 70)

    if total_detections == 0:
        print("❌ Модель не нашла ни одного объекта!")
        print("\n📝 Попробуйте:")
        print("   1. Снизить conf_threshold в train_config.yaml")
        print("   2. Увеличить количество эпох (epochs)")
        print("   3. Увеличить размер изображения (imgsz)")
        print("   4. Использовать большую модель (yolo11s.pt)")
    elif len(not_found) > len(found):
        print("⚠️  Модель находит объекты только на части изображений")
        print("\n📝 Попробуйте:")
        print("   1. Добавить больше размеченных данных")
        print("   2. Увеличить epochs до 300")
        print("   3. Настроить веса потерь (box, cls)")
    else:
        print("✅ Модель работает хорошо!")
        print("\n📝 Для дальнейшего улучшения:")
        print("   1. Добавьте больше размеченных изображений")
        print("   2. Увеличьте conf_threshold для уменьшения false positives")
        print("   3. Попробуйте разные значения lr0 и weight_decay")

    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
