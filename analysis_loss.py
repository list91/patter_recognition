"""
Анализ loss графиков
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Для сохранения без GUI
import matplotlib.pyplot as plt
from pathlib import Path

# Читаем результаты
results_path = Path("runs/detect/quick_train/results.csv")
df = pd.read_csv(results_path)

print("="*80)
print("LOSS ANALYSIS")
print("="*80)

print(f"\nEpochs: {len(df)}")

# Ключевые метрики
print(f"\nTrain losses (first -> last):")
print(f"  box_loss: {df['train/box_loss'].iloc[0]:.2f} -> {df['train/box_loss'].iloc[-1]:.2f}")
print(f"  cls_loss: {df['train/cls_loss'].iloc[0]:.2f} -> {df['train/cls_loss'].iloc[-1]:.2f}")
print(f"  dfl_loss: {df['train/dfl_loss'].iloc[0]:.2f} -> {df['train/dfl_loss'].iloc[-1]:.2f}")

print(f"\nVal losses (first -> last):")
print(f"  box_loss: {df['val/box_loss'].iloc[0]:.2f} -> {df['val/box_loss'].iloc[-1]:.2f}")
print(f"  cls_loss: {df['val/cls_loss'].iloc[0]:.2f} -> {df['val/cls_loss'].iloc[-1]:.2f}")
print(f"  dfl_loss: {df['val/dfl_loss'].iloc[0]:.2f} -> {df['val/dfl_loss'].iloc[-1]:.2f}")

print(f"\nMetrics (always):")
print(f"  Precision: {df['metrics/precision(B)'].iloc[-1]:.3f}")
print(f"  Recall:    {df['metrics/recall(B)'].iloc[-1]:.3f}")
print(f"  mAP50:     {df['metrics/mAP50(B)'].iloc[-1]:.3f}")
print(f"  mAP50-95:  {df['metrics/mAP50-95(B)'].iloc[-1]:.3f}")

# Анализ проблем
print(f"\n{'='*80}")
print("PROBLEMS DETECTED")
print("="*80)

problems = []

# 1. cls_loss зашкаливает
if df['train/cls_loss'].iloc[-1] > 10:
    problems.append(f"[!] train/cls_loss = {df['train/cls_loss'].iloc[-1]:.1f} (too high, model can't classify)")

if df['val/cls_loss'].iloc[-1] > 50:
    problems.append(f"[!] val/cls_loss = {df['val/cls_loss'].iloc[-1]:.1f} (EXTREME overfitting or objects invisible)")

# 2. box_loss растёт
if df['train/box_loss'].iloc[-1] > df['train/box_loss'].iloc[0]:
    problems.append(f"[!] train/box_loss INCREASED ({df['train/box_loss'].iloc[0]:.2f} -> {df['train/box_loss'].iloc[-1]:.2f})")

# 3. Все метрики = 0
if df['metrics/mAP50(B)'].iloc[-1] == 0:
    problems.append(f"[!] ALL metrics = 0 (model detects NOTHING correctly)")

# 4. Нет прогресса
cls_loss_change = abs(df['train/cls_loss'].iloc[-1] - df['train/cls_loss'].iloc[0])
if cls_loss_change < 5:
    problems.append(f"[!] cls_loss barely changed ({cls_loss_change:.1f}), model NOT learning")

if problems:
    for p in problems:
        print(p)
else:
    print("[OK] No obvious problems (but check metrics)")

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Train losses
axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='box_loss')
axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='cls_loss')
axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='dfl_loss')
axes[0, 0].set_title('Train Losses')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Val losses
axes[0, 1].plot(df['epoch'], df['val/box_loss'], label='box_loss')
axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='cls_loss')
axes[0, 1].plot(df['epoch'], df['val/dfl_loss'], label='dfl_loss')
axes[0, 1].set_title('Val Losses')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Metrics
axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
axes[1, 0].set_title('Metrics')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Value')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Learning rate
axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='lr')
axes[1, 1].set_title('Learning Rate')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('LR')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
output_path = Path("results/loss_analysis.png")
plt.savefig(output_path)
print(f"\nGraph saved: {output_path}")
print("="*80)
