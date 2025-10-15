#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для сравнения результатов разных экспериментов.
"""

import json
import sys
from pathlib import Path
import pandas as pd

# Исправление кодировки для Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def load_experiment_metrics(run_dir):
    """Загружает метрики эксперимента."""
    run_path = Path(run_dir)

    if not run_path.exists():
        return None

    # Загружаем финальные метрики
    metrics_path = run_path / 'final_metrics.json'
    experiment_info_path = run_path / 'experiment_info.json'

    if not metrics_path.exists():
        return None

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Загружаем конфигурацию
    config = {}
    if experiment_info_path.exists():
        with open(experiment_info_path, 'r', encoding='utf-8') as f:
            exp_info = json.load(f)
            config = exp_info.get('config', {})

    return {
        'name': run_path.name,
        'metrics': metrics,
        'config': config
    }

def main():
    print("=" * 80)
    print("📊 СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ")
    print("=" * 80 + "\n")

    # Находим все эксперименты
    runs_dir = Path("runs/detect")

    if not runs_dir.exists():
        print("❌ Папка runs/detect не найдена!")
        return

    # Собираем все эксперименты
    experiments = []
    for run_path in sorted(runs_dir.iterdir()):
        if run_path.is_dir():
            exp_data = load_experiment_metrics(run_path)
            if exp_data:
                experiments.append(exp_data)

    if not experiments:
        print("❌ Не найдено ни одного эксперимента с метриками!")
        print("💡 Сначала запустите train.py")
        return

    print(f"✅ Найдено экспериментов: {len(experiments)}\n")

    # Создаём таблицу сравнения
    print("=" * 80)
    print("📋 СРАВНЕНИЕ МЕТРИК")
    print("=" * 80 + "\n")

    # Заголовок
    header = f"{'Эксперимент':<20} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10} {'Box Loss':>10}"
    print(header)
    print("-" * 80)

    # Строки таблицы
    for exp in experiments:
        m = exp['metrics']
        row = (f"{exp['name']:<20} "
               f"{m['precision']:>10.4f} "
               f"{m['recall']:>10.4f} "
               f"{m['mAP50']:>10.4f} "
               f"{m['mAP50-95']:>10.4f} "
               f"{m['train_box_loss']:>10.4f}")
        print(row)

    # Лучшие модели по разным метрикам
    print("\n" + "=" * 80)
    print("🏆 ЛУЧШИЕ МОДЕЛИ")
    print("=" * 80 + "\n")

    best_precision = max(experiments, key=lambda x: x['metrics']['precision'])
    best_recall = max(experiments, key=lambda x: x['metrics']['recall'])
    best_mAP50 = max(experiments, key=lambda x: x['metrics']['mAP50'])
    best_mAP50_95 = max(experiments, key=lambda x: x['metrics']['mAP50-95'])

    print(f"🥇 Лучшая Precision:  {best_precision['name']} ({best_precision['metrics']['precision']:.4f})")
    print(f"🥇 Лучший Recall:     {best_recall['name']} ({best_recall['metrics']['recall']:.4f})")
    print(f"🥇 Лучший mAP50:      {best_mAP50['name']} ({best_mAP50['metrics']['mAP50']:.4f})")
    print(f"🥇 Лучший mAP50-95:   {best_mAP50_95['name']} ({best_mAP50_95['metrics']['mAP50-95']:.4f})")

    # Сравнение конфигураций
    print("\n" + "=" * 80)
    print("⚙️  СРАВНЕНИЕ КОНФИГУРАЦИЙ")
    print("=" * 80 + "\n")

    # Ключевые параметры для сравнения
    key_params = ['model', 'epochs', 'imgsz', 'lr0', 'batch', 'box', 'cls']

    for param in key_params:
        print(f"\n📌 {param}:")
        for exp in experiments:
            value = exp['config'].get(param, 'N/A')
            print(f"   {exp['name']:<20} {value}")

    # Экспорт в CSV
    csv_path = Path("runs/experiments_comparison.csv")

    rows = []
    for exp in experiments:
        row = {
            'name': exp['name'],
            **exp['metrics'],
            **{f'config_{k}': v for k, v in exp['config'].items() if k in key_params}
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print(f"💾 Таблица сравнения сохранена: {csv_path}")
    print("=" * 80 + "\n")

    # Рекомендации
    print("=" * 80)
    print("💡 РЕКОМЕНДАЦИИ")
    print("=" * 80 + "\n")

    # Анализ тренда
    if len(experiments) >= 2:
        # Сортируем по времени (по имени, если есть суффиксы)
        sorted_exps = sorted(experiments, key=lambda x: x['name'])

        first_mAP = sorted_exps[0]['metrics']['mAP50']
        last_mAP = sorted_exps[-1]['metrics']['mAP50']

        if last_mAP > first_mAP:
            improvement = ((last_mAP - first_mAP) / first_mAP) * 100
            print(f"📈 Прогресс: mAP50 улучшился на {improvement:.1f}%")
            print(f"   Первый эксперимент: {first_mAP:.4f}")
            print(f"   Последний эксперимент: {last_mAP:.4f}")
            print("\n✅ Продолжайте экспериментировать в том же направлении!")
        elif last_mAP < first_mAP:
            decline = ((first_mAP - last_mAP) / first_mAP) * 100
            print(f"📉 Ухудшение: mAP50 снизился на {decline:.1f}%")
            print(f"   Первый эксперимент: {first_mAP:.4f}")
            print(f"   Последний эксперимент: {last_mAP:.4f}")
            print("\n⚠️  Вернитесь к конфигурации с лучшими результатами")
        else:
            print("➡️  Метрики не изменились")

    # Советы по улучшению
    print("\n📝 Что попробовать дальше:")

    best_overall = max(experiments, key=lambda x: x['metrics']['mAP50'])

    if best_overall['metrics']['mAP50'] < 0.3:
        print("   ❌ Низкое качество (<0.3 mAP50):")
        print("      1. Добавьте больше размеченных данных (минимум 50-100 изображений)")
        print("      2. Увеличьте epochs до 300")
        print("      3. Попробуйте yolo11s.pt или yolo11m.pt")
    elif best_overall['metrics']['mAP50'] < 0.6:
        print("   ⚠️  Среднее качество (0.3-0.6 mAP50):")
        print("      1. Увеличьте imgsz до 1536")
        print("      2. Настройте веса потерь: box=10.0, cls=1.0")
        print("      3. Попробуйте lr0=0.001 для более стабильного обучения")
    else:
        print("   ✅ Хорошее качество (>0.6 mAP50):")
        print("      1. Увеличьте conf_threshold для уменьшения false positives")
        print("      2. Добавьте валидационный набор")
        print("      3. Попробуйте ансамбль из нескольких моделей")

    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
