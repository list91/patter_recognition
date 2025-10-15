"""
Диагностика: показывает КАК выглядит train_01.jpg после resize до разных imgsz
"""
import cv2
import numpy as np
from pathlib import Path

def letterbox_resize(img, new_size=640):
    """Resize как в YOLO - с сохранением пропорций"""
    h, w = img.shape[:2]

    # Вычисляем масштаб
    scale = min(new_size / h, new_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding до квадрата
    top = (new_size - new_h) // 2
    bottom = new_size - new_h - top
    left = (new_size - new_w) // 2
    right = new_size - new_w - left

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img_padded, scale, (left, top)

def draw_boxes_on_resized(img_path, label_path, imgsz, output_path):
    """Отрисовывает bbox на resized изображении"""
    # Загрузка
    img = cv2.imread(str(img_path))
    orig_h, orig_w = img.shape[:2]

    # Resize как YOLO
    img_resized, scale, (pad_x, pad_y) = letterbox_resize(img, imgsz)

    # Чтение разметки
    with open(label_path, 'r') as f:
        lines = f.readlines()

    boxes_info = []

    # Отрисовка bbox
    for line in lines:
        cls, x_c, y_c, w, h = map(float, line.strip().split())

        # Конвертация из YOLO формата в pixel координаты (оригинал)
        x_c_px = x_c * orig_w
        y_c_px = y_c * orig_h
        w_px = w * orig_w
        h_px = h * orig_h

        # Применяем scale и padding
        x_c_scaled = x_c_px * scale + pad_x
        y_c_scaled = y_c_px * scale + pad_y
        w_scaled = w_px * scale
        h_scaled = h_px * scale

        # Bounding box corners
        x1 = int(x_c_scaled - w_scaled / 2)
        y1 = int(y_c_scaled - h_scaled / 2)
        x2 = int(x_c_scaled + w_scaled / 2)
        y2 = int(y_c_scaled + h_scaled / 2)

        # Отрисовка
        cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Подпись с размерами
        box_w = x2 - x1
        box_h = y2 - y1
        label = f"{box_w}x{box_h}px"
        cv2.putText(img_resized, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        boxes_info.append((box_w, box_h))

    # Сохранение
    cv2.imwrite(str(output_path), img_resized)

    return boxes_info, (orig_w, orig_h), imgsz

# Тестируем разные размеры
img_path = Path("data/images/train/train_01.jpg")
label_path = Path("data/labels/train/train_01.txt")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

print("=" * 80)
print("ДИАГНОСТИКА РАЗМЕРОВ ОБЪЕКТОВ")
print("=" * 80)

for imgsz in [416, 640, 920, 1280, 1920]:
    output_path = results_dir / f"resize_debug_{imgsz}.jpg"

    boxes_info, (orig_w, orig_h), size = draw_boxes_on_resized(
        img_path, label_path, imgsz, output_path
    )

    print(f"\nimgsz={imgsz} (original: {orig_w}x{orig_h})")
    print(f"   Saved: {output_path}")

    # Статистика по размерам bbox
    widths = [w for w, h in boxes_info]
    heights = [h for w, h in boxes_info]

    print(f"   BBox sizes after resize:")
    print(f"   - Width:  min={min(widths):.1f}px, max={max(widths):.1f}px, avg={np.mean(widths):.1f}px")
    print(f"   - Height: min={min(heights):.1f}px, max={max(heights):.1f}px, avg={np.mean(heights):.1f}px")

    # Критичность
    min_side = min(min(widths), min(heights))
    if min_side < 10:
        print(f"   [!] CRITICAL: min side {min_side:.1f}px < 10px!")
    elif min_side < 15:
        print(f"   [!] RISKY: min side {min_side:.1f}px < 15px")
    else:
        print(f"   [OK] Good: min side {min_side:.1f}px >= 15px")

print("\n" + "=" * 80)
print("All visualizations saved to results/")
print("=" * 80)
