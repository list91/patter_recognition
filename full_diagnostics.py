"""
ПОЛНАЯ ДИАГНОСТИКА: Проверка ВСЕХ возможных проблем обучения
"""
import cv2
import numpy as np
from pathlib import Path
import yaml

print("=" * 80)
print("FULL DIAGNOSTICS: ALL POTENTIAL ISSUES")
print("=" * 80)

# Пути
img_path = Path("data/images/train/train_01.jpg")
label_path = Path("data/labels/train/train_01.txt")
config_path = Path("train_config.yaml")

# ============================================================================
# 1. ПРОВЕРКА РАЗМЕТКИ
# ============================================================================
print("\n" + "=" * 80)
print("1. LABEL FILE VALIDATION")
print("=" * 80)

# Читаем изображение для получения размеров
img = cv2.imread(str(img_path))
img_h, img_w = img.shape[:2]
print(f"Image size: {img_w}x{img_h}")

# Читаем разметку
with open(label_path, 'r') as f:
    labels = [line.strip().split() for line in f.readlines()]

print(f"Total annotations: {len(labels)}")

# Проверка каждой аннотации
issues = []
for i, label in enumerate(labels):
    if len(label) != 5:
        issues.append(f"  [!] Line {i+1}: Wrong format (expected 5 values, got {len(label)})")
        continue

    try:
        cls, x_c, y_c, w, h = map(float, label)
    except ValueError:
        issues.append(f"  [!] Line {i+1}: Non-numeric values")
        continue

    # Проверка диапазонов
    if not (0 <= x_c <= 1):
        issues.append(f"  [!] Line {i+1}: x_center={x_c:.4f} out of range [0,1]")
    if not (0 <= y_c <= 1):
        issues.append(f"  [!] Line {i+1}: y_center={y_c:.4f} out of range [0,1]")
    if not (0 < w <= 1):
        issues.append(f"  [!] Line {i+1}: width={w:.6f} out of range (0,1]")
    if not (0 < h <= 1):
        issues.append(f"  [!] Line {i+1}: height={h:.6f} out of range (0,1]")

    # Проверка что bbox не выходит за границы
    x1 = x_c - w/2
    x2 = x_c + w/2
    y1 = y_c - h/2
    y2 = y_c + h/2

    if x1 < 0 or x2 > 1 or y1 < 0 or y2 > 1:
        issues.append(f"  [!] Line {i+1}: BBox goes outside image bounds")

    # Проверка класса
    if int(cls) != 0:
        issues.append(f"  [!] Line {i+1}: Class={int(cls)} (expected 0 for 'switch')")

if issues:
    print("\n[!] ISSUES FOUND:")
    for issue in issues:
        print(issue)
else:
    print("\n[OK] All annotations are valid!")

# Статистика по размерам объектов (в пикселях)
widths_px = []
heights_px = []
areas_px = []

for label in labels:
    cls, x_c, y_c, w, h = map(float, label)
    w_px = w * img_w
    h_px = h * img_h
    widths_px.append(w_px)
    heights_px.append(h_px)
    areas_px.append(w_px * h_px)

print(f"\nObject sizes (pixels):")
print(f"  Width:  min={min(widths_px):.1f}, max={max(widths_px):.1f}, avg={np.mean(widths_px):.1f}")
print(f"  Height: min={min(heights_px):.1f}, max={max(heights_px):.1f}, avg={np.mean(heights_px):.1f}")
print(f"  Area:   min={min(areas_px):.0f}, max={max(areas_px):.0f}, avg={np.mean(areas_px):.0f} px^2")

# Относительные размеры
rel_sizes = [(w*img_w * h*img_h) / (img_w * img_h) for _, _, _, w, h in [map(float, l) for l in labels]]
print(f"  Relative area: {np.mean(rel_sizes)*100:.6f}% of image")

# ============================================================================
# 2. ПРОВЕРКА ИЗОБРАЖЕНИЯ
# ============================================================================
print("\n" + "=" * 80)
print("2. IMAGE VALIDATION")
print("=" * 80)

print(f"File exists: {img_path.exists()}")
print(f"Image shape: {img.shape} (H, W, C)")
print(f"Image dtype: {img.dtype}")
print(f"Image range: [{img.min()}, {img.max()}]")

# Проверка на повреждения
if img is None:
    print("[!] ERROR: Image cannot be read!")
elif img.size == 0:
    print("[!] ERROR: Image is empty!")
else:
    print("[OK] Image is valid")

# Статистика по цветам
mean_color = img.mean(axis=(0, 1))
std_color = img.std(axis=(0, 1))
print(f"Mean color (BGR): [{mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f}]")
print(f"Std color (BGR):  [{std_color[0]:.1f}, {std_color[1]:.1f}, {std_color[2]:.1f}]")

# ============================================================================
# 3. АНАЛИЗ АУГМЕНТАЦИЙ
# ============================================================================
print("\n" + "=" * 80)
print("3. AUGMENTATION ANALYSIS")
print("=" * 80)

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

aug_params = {
    'mosaic': config.get('mosaic', 0),
    'degrees': config.get('degrees', 0),
    'translate': config.get('translate', 0),
    'scale': config.get('scale', 0),
    'fliplr': config.get('fliplr', 0),
    'flipud': config.get('flipud', 0),
    'mixup': config.get('mixup', 0),
    'copy_paste': config.get('copy_paste', 0),
}

print("\nGeometric augmentations:")
for key, value in aug_params.items():
    status = "[ENABLED]" if value > 0 else "[DISABLED]"
    print(f"  {key:12} = {value:5} {status}")

# Проверка проблемных аугментаций для мелких объектов
warnings = []
if config.get('mosaic', 0) > 0.5:
    warnings.append("  [!] mosaic>0.5: May destroy tiny objects when stitching")
if config.get('scale', 0) > 0.3:
    warnings.append("  [!] scale>0.3: May make small objects even smaller")
if config.get('mixup', 0) > 0:
    warnings.append("  [!] mixup>0: May create confusing overlaps for tiny objects")

if warnings:
    print("\n[!] POTENTIAL ISSUES:")
    for w in warnings:
        print(w)
else:
    print("\n[OK] Augmentation settings look safe for small objects")

# ============================================================================
# 4. АНАЛИЗ LOSS WEIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("4. LOSS WEIGHTS ANALYSIS")
print("=" * 80)

box_loss = config.get('box', 7.5)
cls_loss = config.get('cls', 0.5)
dfl_loss = config.get('dfl', 1.5)

total = box_loss + cls_loss + dfl_loss
print(f"Loss weights:")
print(f"  box: {box_loss:5.1f} ({box_loss/total*100:5.1f}%)")
print(f"  cls: {cls_loss:5.1f} ({cls_loss/total*100:5.1f}%)")
print(f"  dfl: {dfl_loss:5.1f} ({dfl_loss/total*100:5.1f}%)")

# Рекомендации
if box_loss / cls_loss < 5:
    print("\n[!] box/cls ratio < 5: Consider increasing 'box' for better localization")
if cls_loss < 0.3:
    print("[!] cls < 0.3: Very low, may struggle with classification")

# ============================================================================
# 5. ПРОВЕРКА ГИПЕРПАРАМЕТРОВ
# ============================================================================
print("\n" + "=" * 80)
print("5. HYPERPARAMETERS CHECK")
print("=" * 80)

lr0 = config.get('lr0', 0.01)
lrf = config.get('lrf', 0.01)
momentum = config.get('momentum', 0.937)
weight_decay = config.get('weight_decay', 0.0005)
epochs = config.get('epochs', 100)
imgsz = config.get('imgsz', 640)
batch = config.get('batch', 16)

print(f"Learning rate:")
print(f"  lr0:          {lr0}")
print(f"  lrf:          {lrf}")
print(f"  final_lr:     {lr0 * lrf:.6f}")

print(f"\nOptimization:")
print(f"  optimizer:    {config.get('optimizer', 'SGD')}")
print(f"  momentum:     {momentum}")
print(f"  weight_decay: {weight_decay}")

print(f"\nTraining:")
print(f"  epochs:       {epochs}")
print(f"  imgsz:        {imgsz}")
print(f"  batch:        {batch}")

# Проверка на потенциальные проблемы
hyper_warnings = []
if lr0 > 0.01:
    hyper_warnings.append("  [!] lr0 > 0.01: May be too high, causing instability")
if lr0 < 0.0001:
    hyper_warnings.append("  [!] lr0 < 0.0001: May be too low, slow convergence")
if epochs < 50:
    hyper_warnings.append("  [!] epochs < 50: May be too few for small objects")
if batch > 4 and imgsz > 1280:
    hyper_warnings.append("  [!] batch>4 with large imgsz: Memory issues possible")

if hyper_warnings:
    print("\n[!] POTENTIAL ISSUES:")
    for w in hyper_warnings:
        print(w)

# ============================================================================
# 6. ANCHOR SIZE ESTIMATION
# ============================================================================
print("\n" + "=" * 80)
print("6. ANCHOR SIZE ESTIMATION")
print("=" * 80)

# Вычисляем размеры объектов после resize
scale_factor = imgsz / max(img_w, img_h)
scaled_widths = [w * scale_factor for w in widths_px]
scaled_heights = [h * scale_factor for h in heights_px]

print(f"Object sizes after resize to {imgsz}:")
print(f"  Width:  min={min(scaled_widths):.1f}, max={max(scaled_widths):.1f}, avg={np.mean(scaled_widths):.1f} px")
print(f"  Height: min={min(scaled_heights):.1f}, max={max(scaled_heights):.1f}, avg={np.mean(scaled_heights):.1f} px")

# Рекомендуемые anchor sizes (примерная оценка)
avg_w = np.mean(scaled_widths)
avg_h = np.mean(scaled_heights)
print(f"\nEstimated optimal anchor size: ~{avg_w:.1f}x{avg_h:.1f} px")

# Проверка размеров для YOLO heads (P3, P4, P5)
# P3: stride=8  (для мелких объектов)
# P4: stride=16 (для средних объектов)
# P5: stride=32 (для крупных объектов)
print(f"\nWhich YOLO head should detect these objects:")
if avg_w < 16 or avg_h < 16:
    print(f"  -> P3 head (stride=8, для мелких объектов < 16px)")
    print(f"     [!] Objects are VERY SMALL, may be at detection limit!")
elif avg_w < 32 or avg_h < 32:
    print(f"  -> P3/P4 heads (stride=8-16, для мелких/средних объектов)")
else:
    print(f"  -> P4/P5 heads (stride=16-32, для средних/крупных объектов)")

# ============================================================================
# 7. ВЫЧИСЛЕНИЕ МЕТРИК НА ТРЕНИРОВОЧНОМ НАБОРЕ
# ============================================================================
print("\n" + "=" * 80)
print("7. THEORETICAL DETECTION DIFFICULTY")
print("=" * 80)

# Оценка сложности детекции
min_side = min(min(scaled_widths), min(scaled_heights))
difficulty_score = 0

if min_side < 5:
    difficulty = "EXTREME"
    difficulty_score = 5
elif min_side < 10:
    difficulty = "VERY HIGH"
    difficulty_score = 4
elif min_side < 20:
    difficulty = "HIGH"
    difficulty_score = 3
elif min_side < 40:
    difficulty = "MODERATE"
    difficulty_score = 2
else:
    difficulty = "LOW"
    difficulty_score = 1

print(f"Detection difficulty: {difficulty} (score: {difficulty_score}/5)")
print(f"  Min side:     {min_side:.1f} px")
print(f"  Image size:   {imgsz} px")
print(f"  Training set: 1 image only (EXTREME overfitting risk)")

# ============================================================================
# ИТОГОВЫЕ РЕКОМЕНДАЦИИ
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

recommendations = []

# 1. Размер изображения
if min_side < 10:
    recommended_imgsz = int(imgsz * (15 / min_side))
    recommendations.append(f"1. INCREASE imgsz: {recommended_imgsz}+ (currently {imgsz}, objects {min_side:.1f}px)")

# 2. Аугментации
if config.get('mosaic', 0) > 0:
    recommendations.append(f"2. TRY disabling mosaic: set mosaic=0.0 (may destroy tiny objects)")

# 3. Loss weights
if box_loss / cls_loss < 10:
    recommendations.append(f"3. INCREASE box loss: set box=15.0 (currently {box_loss})")

# 4. Epochs
if epochs < 100:
    recommendations.append(f"4. INCREASE epochs: 100-200 (currently {epochs})")

# 5. Learning rate
if lr0 > 0.001:
    recommendations.append(f"5. DECREASE lr0: 0.0005-0.001 (currently {lr0})")

# 6. Модель
recommendations.append(f"6. TRY larger model: yolov8m.pt or yolov8l.pt (currently {config.get('model', '?')})")

# 7. Данные
recommendations.append(f"7. ADD more training data: 1 image is NOT enough!")

print("\nTop recommendations:")
for rec in recommendations[:5]:
    print(f"  {rec}")

print("\n" + "=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)
