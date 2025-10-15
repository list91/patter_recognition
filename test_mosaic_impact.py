"""
Проверка: КАК mosaic аугментация влияет на мелкие объекты
"""
import cv2
import numpy as np
from pathlib import Path

def create_mosaic_simulation(img_path, imgsz=1920):
    """Симулирует mosaic аугментацию"""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Resize как YOLO (letterbox)
    scale = min(imgsz / h, imgsz / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Padding
    top = (imgsz - new_h) // 2
    left = (imgsz - new_w) // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, imgsz-new_h-top,
                                     left, imgsz-new_w-left,
                                     cv2.BORDER_CONSTANT, value=(114,114,114))

    # Mosaic: склейка 4 изображений
    mosaic_size = imgsz * 2
    mosaic = np.zeros((mosaic_size, mosaic_size, 3), dtype=np.uint8)

    # Размещаем 4 копии изображения
    mosaic[0:imgsz, 0:imgsz] = img_padded
    mosaic[0:imgsz, imgsz:mosaic_size] = img_padded
    mosaic[imgsz:mosaic_size, 0:imgsz] = img_padded
    mosaic[imgsz:mosaic_size, imgsz:mosaic_size] = img_padded

    # Crop центральную часть (как YOLO делает)
    center_x, center_y = mosaic_size // 2, mosaic_size // 2
    crop_x1 = center_x - imgsz // 2
    crop_y1 = center_y - imgsz // 2
    crop_x2 = crop_x1 + imgsz
    crop_y2 = crop_y1 + imgsz

    mosaic_cropped = mosaic[crop_y1:crop_y2, crop_x1:crop_x2]

    return img_padded, mosaic_cropped, scale

# Тест
img_path = Path("data/images/train/train_01.jpg")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

print("="*80)
print("MOSAIC AUGMENTATION IMPACT TEST")
print("="*80)

for imgsz in [920, 1280, 1920]:
    print(f"\nimgsz={imgsz}")

    # Без mosaic
    img_normal, mosaic_img, scale = create_mosaic_simulation(img_path, imgsz)

    # Сохраняем
    cv2.imwrite(str(results_dir / f"normal_{imgsz}.jpg"), img_normal)
    cv2.imwrite(str(results_dir / f"mosaic_{imgsz}.jpg"), mosaic_img)

    # Вычисляем размеры объектов
    # Оригинальные объекты: 22×47 px
    orig_w, orig_h = 22, 47

    # После resize
    normal_w = orig_w * scale
    normal_h = orig_h * scale

    # После mosaic (объекты сжимаются в 2 раза + crop может их резать)
    mosaic_w = normal_w / 2
    mosaic_h = normal_h / 2

    print(f"  Normal resize: {normal_w:.1f}x{normal_h:.1f} px")
    print(f"  After mosaic:  {mosaic_w:.1f}x{mosaic_h:.1f} px  [PROBLEM!]")

    if mosaic_w < 5 or mosaic_h < 5:
        print(f"  [!] CRITICAL: Objects < 5px after mosaic!")
    elif mosaic_w < 10 or mosaic_h < 10:
        print(f"  [!] WARNING: Objects < 10px after mosaic!")

print(f"\nVisualizations saved to results/")
print("="*80)
