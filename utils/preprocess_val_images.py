"""
Скрипт для предобработки валидационных изображений.
Закрашивает размеченные объекты средним оттенком самых светлых пикселей.
"""

import os
from PIL import Image
import numpy as np
from datetime import datetime
from pathlib import Path


def get_lightest_pixels_color(image, num_pixels=3):
    """
    Находит N самых светлых пикселей и возвращает их средний цвет.

    Args:
        image: PIL Image объект
        num_pixels: количество самых светлых пикселей для усреднения

    Returns:
        tuple: средний цвет в формате (R, G, B)
    """
    # Конвертируем в RGB если нужно
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Преобразуем в numpy array
    img_array = np.array(image)

    # Вычисляем яркость для каждого пикселя (средняя по RGB)
    brightness = img_array.mean(axis=2)

    # Находим индексы N самых светлых пикселей
    flat_brightness = brightness.flatten()
    lightest_indices = np.argpartition(flat_brightness, -num_pixels)[-num_pixels:]

    # Получаем координаты этих пикселей
    height, width = brightness.shape
    y_coords = lightest_indices // width
    x_coords = lightest_indices % width

    # Получаем RGB значения этих пикселей
    lightest_pixels = img_array[y_coords, x_coords]

    # Вычисляем средний цвет
    avg_color = lightest_pixels.mean(axis=0).astype(np.uint8)

    return tuple(avg_color)


def read_yolo_annotations(label_path):
    """
    Читает YOLO аннотации из файла.

    Args:
        label_path: путь к .txt файлу с аннотациями

    Returns:
        list: список bbox в формате [(class_id, x_center, y_center, width, height), ...]
    """
    annotations = []

    if not os.path.exists(label_path):
        return annotations

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append((class_id, x_center, y_center, width, height))

    return annotations


def yolo_to_pixel_coords(annotation, img_width, img_height):
    """
    Конвертирует YOLO нормализованные координаты в пиксельные.

    Args:
        annotation: tuple (class_id, x_center, y_center, width, height) - нормализованные
        img_width: ширина изображения в пикселях
        img_height: высота изображения в пикселях

    Returns:
        tuple: (x_min, y_min, x_max, y_max) в пикселях
    """
    class_id, x_center, y_center, width, height = annotation

    # Конвертируем центр в пиксели
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height

    # Конвертируем размеры в пиксели
    width_px = width * img_width
    height_px = height * img_height

    # Вычисляем углы bbox
    x_min = int(x_center_px - width_px / 2)
    y_min = int(y_center_px - height_px / 2)
    x_max = int(x_center_px + width_px / 2)
    y_max = int(y_center_px + height_px / 2)

    return (x_min, y_min, x_max, y_max)


def fill_bboxes_with_color(image, annotations, fill_color):
    """
    Закрашивает области bbox указанным цветом.

    Args:
        image: PIL Image объект
        annotations: список YOLO аннотаций
        fill_color: цвет для заливки (R, G, B)

    Returns:
        PIL Image: изображение с закрашенными областями
    """
    # Создаем копию изображения
    result = image.copy()

    # Конвертируем в RGB если нужно
    if result.mode == 'RGBA':
        result = result.convert('RGB')
    elif result.mode != 'RGB':
        result = result.convert('RGB')

    # Преобразуем в numpy для быстрой работы
    img_array = np.array(result)
    img_width, img_height = result.size

    # Закрашиваем каждый bbox
    for annotation in annotations:
        x_min, y_min, x_max, y_max = yolo_to_pixel_coords(annotation, img_width, img_height)

        # Проверяем границы
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)

        # Закрашиваем область
        img_array[y_min:y_max, x_min:x_max] = fill_color

    # Конвертируем обратно в PIL Image
    return Image.fromarray(img_array)


def process_validation_images(val_images_dir, val_labels_dir, output_dir):
    """
    Обрабатывает все валидационные изображения.

    Args:
        val_images_dir: путь к директории с изображениями
        val_labels_dir: путь к директории с аннотациями
        output_dir: путь для сохранения результатов
    """
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)

    # Получаем список всех изображений
    image_files = [f for f in os.listdir(val_images_dir)
                   if f.endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Найдено {len(image_files)} изображений для обработки\n")

    for img_file in image_files:
        print(f"Обработка: {img_file}")

        # Пути к файлам
        img_path = os.path.join(val_images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(val_labels_dir, label_file)

        # Загружаем изображение
        image = Image.open(img_path)
        print(f"  Размер: {image.size}, Режим: {image.mode}")

        # Находим самые светлые пиксели
        lightest_color = get_lightest_pixels_color(image, num_pixels=3)
        print(f"  Средний цвет 3 самых светлых пикселей: RGB{lightest_color}")

        # Читаем аннотации
        annotations = read_yolo_annotations(label_path)
        print(f"  Найдено объектов: {len(annotations)}")

        if len(annotations) == 0:
            print(f"  [!] Нет аннотаций для {img_file}, пропускаем")
            continue

        # Закрашиваем объекты
        processed_image = fill_bboxes_with_color(image, annotations, lightest_color)

        # Сохраняем результат
        output_path = os.path.join(output_dir, img_file)
        # Сохраняем как JPEG если исходник был JPEG
        if img_file.lower().endswith('.jpg') or img_file.lower().endswith('.jpeg'):
            processed_image.save(output_path, 'JPEG', quality=95)
        else:
            processed_image.save(output_path)

        print(f"  [OK] Сохранено: {output_path}\n")

    print(f"Готово! Все результаты в: {output_dir}")


def main():
    """Главная функция."""
    # Пути
    project_dir = Path(__file__).parent.parent
    val_images_dir = project_dir / 'data' / 'yolo_dataset' / 'val' / 'images'
    val_labels_dir = project_dir / 'data' / 'yolo_dataset' / 'val' / 'labels'

    # Создаем директорию для результатов с timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = project_dir / 'results' / f'preprocessed_{timestamp}'

    print("="*60)
    print("Предобработка валидационных изображений")
    print("="*60)
    print(f"Исходные изображения: {val_images_dir}")
    print(f"Аннотации: {val_labels_dir}")
    print(f"Результаты: {output_dir}")
    print("="*60 + "\n")

    # Обработка
    process_validation_images(
        str(val_images_dir),
        str(val_labels_dir),
        str(output_dir)
    )


if __name__ == '__main__':
    main()
