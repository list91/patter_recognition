"""
Тест pretrained yolov8s.pt БЕЗ обучения на train_01.jpg
"""
from ultralytics import YOLO
from pathlib import Path

print("="*70)
print("TEST PRETRAINED YOLO (NO TRAINING)")
print("="*70)

# Загружаем pretrained модель
model = YOLO("yolov8s.pt")

# Тестируем на train_01.jpg
img_path = "data/images/train/train_01.jpg"

print(f"\nTesting on: {img_path}")
print(f"Image size: Will use default (640)")
print(f"Confidence threshold: 0.001 (very low)")

# Предсказание
results = model.predict(
    source=img_path,
    conf=0.001,
    iou=0.45,
    save=False,
    verbose=False
)

result = results[0]
boxes = result.boxes

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"Total detections: {len(boxes)}")

if len(boxes) > 0:
    # Группировка по классам
    class_counts = {}
    conf_values = []

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = model.names[cls]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        conf_values.append(conf)

    print(f"\nDetected classes:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {count} objects")

    print(f"\nConfidence stats:")
    print(f"  Min: {min(conf_values):.3f}")
    print(f"  Max: {max(conf_values):.3f}")
    print(f"  Avg: {sum(conf_values)/len(conf_values):.3f}")
else:
    print("\nNO DETECTIONS FOUND!")
    print("Pretrained model sees NOTHING on this image.")

print(f"{'='*70}\n")
