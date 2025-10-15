#!/usr/bin/env python3
"""
Скрипт обучения YOLO11 с поддержкой конфигурации и детальным мониторингом.
"""

import yaml
import sys
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

def load_config(config_path="train_config.yaml"):
    """Загружает конфигурацию из YAML файла."""
    if not Path(config_path).exists():
        print(f"⚠️  Конфиг {config_path} не найден, используем параметры по умолчанию")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_experiment_info(config, results_dir):
    """Сохраняет информацию об эксперименте."""
    info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'results_dir': str(results_dir)
    }

    info_path = Path(results_dir) / 'experiment_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"💾 Информация об эксперименте сохранена: {info_path}")

def print_config(config):
    """Красиво выводит конфигурацию."""
    print("\n" + "=" * 70)
    print(f"🔧 КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА: {config.get('experiment_name', 'default')}")
    print("=" * 70)

    categories = {
        'Основные': ['model', 'epochs', 'imgsz', 'batch'],
        'Оптимизатор': ['optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay'],
        'Цветовые аугментации': ['hsv_h', 'hsv_s', 'hsv_v'],
        'Геометрические аугментации': ['mosaic', 'degrees', 'translate', 'scale', 'fliplr', 'flipud'],
        'Потери': ['box', 'cls', 'dfl'],
        'Тестирование': ['conf_threshold', 'iou_threshold'],
    }

    for category, keys in categories.items():
        print(f"\n📋 {category}:")
        for key in keys:
            if key in config:
                print(f"   {key:20} = {config[key]}")

    print("=" * 70 + "\n")

def main():
    # Загружаем конфигурацию
    config_file = sys.argv[1] if len(sys.argv) > 1 else "train_config.yaml"
    config = load_config(config_file)

    # Параметры по умолчанию
    defaults = {
        'experiment_name': 'baseline',
        'model': 'yolo11n.pt',
        'epochs': 150,
        'imgsz': 1280,
        'batch': 2,
        'optimizer': 'auto',
        'lr0': 0.002,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'mosaic': 0.0,
        'degrees': 0.0,
        'translate': 0.0,
        'scale': 0.0,
        'fliplr': 0.0,
        'flipud': 0.0,
        'shear': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.5,
        'hsv_v': 0.4,
        'dropout': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'patience': 50,
        'close_mosaic': 10,
        'warmup_epochs': 3.0,
        'cos_lr': False,
        'conf_threshold': 0.1,
        'iou_threshold': 0.45,
    }

    # Объединяем с дефолтами
    for key, value in defaults.items():
        config.setdefault(key, value)

    print_config(config)

    print("🚀 Начинаем обучение модели YOLO11 для детекции switch")
    print(f"📂 Используется конфигурация: {config_file}\n")

    # Загружаем модель
    model = YOLO(config['model'])

    # Формируем имя эксперимента
    exp_name = f"{config['experiment_name']}"

    # Запускаем обучение
    print("⏳ Обучение началось...\n")
    results = model.train(
        data="data.yaml",

        # Основные параметры
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],

        # Оптимизатор
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],

        # Геометрические аугментации
        mosaic=config['mosaic'],
        degrees=config['degrees'],
        translate=config['translate'],
        scale=config['scale'],
        fliplr=config['fliplr'],
        flipud=config['flipud'],
        shear=config['shear'],

        # Цветовые аугментации
        hsv_h=config['hsv_h'],
        hsv_s=config['hsv_s'],
        hsv_v=config['hsv_v'],

        # Dropout и другие аугментации
        dropout=config['dropout'],
        mixup=config['mixup'],
        copy_paste=config['copy_paste'],

        # Веса потерь
        box=config['box'],
        cls=config['cls'],
        dfl=config['dfl'],

        # Другие параметры
        patience=config['patience'],
        close_mosaic=config['close_mosaic'],
        warmup_epochs=config['warmup_epochs'],
        cos_lr=config['cos_lr'],

        # Сохранение
        save=True,
        save_period=10,
        device="cpu",
        workers=2,
        project="runs/detect",
        name=exp_name,
        exist_ok=False,  # Создаст новую папку с суффиксом
        verbose=True,
        plots=True,
    )

    # Получаем путь к результатам
    results_dir = Path(model.trainer.save_dir)

    print("\n" + "=" * 70)
    print("✅ Обучение завершено!")
    print("=" * 70)
    print(f"📁 Папка с результатами: {results_dir}")
    print(f"🏆 Лучшая модель: {results_dir}/weights/best.pt")
    print(f"📊 Графики: {results_dir}/results.png")

    # Сохраняем информацию об эксперименте
    save_experiment_info(config, results_dir)

    # Выводим финальные метрики
    print("\n" + "=" * 70)
    print("📈 ФИНАЛЬНЫЕ МЕТРИКИ")
    print("=" * 70)

    try:
        import pandas as pd
        results_csv = results_dir / "results.csv"
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            last_row = df.iloc[-1]

            print(f"\n📊 Эпоха {int(last_row['epoch'])}:")
            print(f"   Train Loss (box): {last_row['train/box_loss']:.4f}")
            print(f"   Train Loss (cls): {last_row['train/cls_loss']:.4f}")
            print(f"   Val Loss (box):   {last_row['val/box_loss']:.4f}")
            print(f"   Val Loss (cls):   {last_row['val/cls_loss']:.4f}")
            print(f"\n🎯 Метрики качества:")
            print(f"   Precision:  {last_row['metrics/precision(B)']:.4f}")
            print(f"   Recall:     {last_row['metrics/recall(B)']:.4f}")
            print(f"   mAP50:      {last_row['metrics/mAP50(B)']:.4f}")
            print(f"   mAP50-95:   {last_row['metrics/mAP50-95(B)']:.4f}")

            # Сохраняем метрики
            metrics = {
                'precision': float(last_row['metrics/precision(B)']),
                'recall': float(last_row['metrics/recall(B)']),
                'mAP50': float(last_row['metrics/mAP50(B)']),
                'mAP50-95': float(last_row['metrics/mAP50-95(B)']),
                'train_box_loss': float(last_row['train/box_loss']),
                'train_cls_loss': float(last_row['train/cls_loss']),
            }

            metrics_path = results_dir / 'final_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

    except Exception as e:
        print(f"⚠️  Не удалось вывести финальные метрики: {e}")

    print("\n" + "=" * 70)
    print("💡 СЛЕДУЮЩИЕ ШАГИ:")
    print("=" * 70)
    print(f"1. Запустите тестирование:")
    print(f"   python test_and_visualize.py {exp_name}")
    print(f"\n2. Сравните с другими экспериментами:")
    print(f"   python compare_runs.py")
    print(f"\n3. Измените параметры в train_config.yaml и запустите снова")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
