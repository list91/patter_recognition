"""
Генератор YOLO датасета на базе реальных фонов (предобработанных схем).
Использует очищенные от объектов реальные схемы как фоны.
"""

import os
import random
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter


def create_grain_mask(height, width, num_spots_range=(10, 20), spot_size_range=(50, 200), blur_sigma=50):
    """
    Создает маску со случайными пятнами для grain эффекта.

    Args:
        height: высота изображения
        width: ширина изображения
        num_spots_range: диапазон количества пятен
        spot_size_range: диапазон размера пятен
        blur_sigma: сигма для гауссова размытия

    Returns:
        Нормализованная маска со значениями от 0 до 1
    """
    mask = np.zeros((height, width), dtype=np.float32)

    num_spots = np.random.randint(num_spots_range[0], num_spots_range[1] + 1)

    for _ in range(num_spots):
        center_y = np.random.randint(0, height)
        center_x = np.random.randint(0, width)
        base_radius = np.random.randint(spot_size_range[0], spot_size_range[1])

        aspect_ratio = np.random.uniform(0.4, 1.5)
        angle = np.random.uniform(0, 2 * np.pi)

        y, x = np.ogrid[:height, :width]
        y_shifted = y - center_y
        x_shifted = x - center_x

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rot = cos_angle * x_shifted + sin_angle * y_shifted
        y_rot = -sin_angle * x_shifted + cos_angle * y_shifted

        radius_x = base_radius
        radius_y = base_radius * aspect_ratio

        distance = np.sqrt((x_rot / radius_x)**2 + (y_rot / radius_y)**2)

        deformation = np.random.uniform(0.85, 1.15)
        threshold = 1.0 * deformation

        spot_intensity = np.random.uniform(0.3, 1.0)
        spot = (distance <= threshold).astype(np.float32) * spot_intensity

        if np.random.random() > 0.5:
            noise_scale = np.random.uniform(0.05, 0.15)
            spot = spot * (1 - noise_scale + noise_scale * 2 * np.random.random(spot.shape))

        mask = np.maximum(mask, spot)

    mask = gaussian_filter(mask, sigma=blur_sigma)

    if mask.max() > 0:
        mask = mask / mask.max()

    return mask


def apply_grain(image, grain_intensity):
    """
    Применяет grain эффект к изображению.

    Args:
        image: PIL Image
        grain_intensity: интенсивность grain (0-100)

    Returns:
        PIL Image с примененным grain
    """
    if grain_intensity == 0:
        return image

    img_array = np.array(image).astype(np.float32)
    height, width = img_array.shape[:2]

    mask = create_grain_mask(height, width, num_spots_range=(10, 18),
                            spot_size_range=(80, 250), blur_sigma=60)

    if len(img_array.shape) == 3:
        mask = mask[:, :, np.newaxis]

    noise = np.random.normal(0, grain_intensity, img_array.shape)
    noisy_img = img_array + (noise * mask)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_img)


def create_transparency_mask(obj_image, threshold=240):
    """
    Создает маску прозрачности для объекта на белом фоне.

    Args:
        obj_image: PIL Image объекта
        threshold: порог яркости для определения фона (0-255)

    Returns:
        PIL Image маска в режиме 'L'
    """
    obj_array = np.array(obj_image)
    # Берем минимальную яркость по RGB каналам
    brightness = obj_array.min(axis=2)
    # Темные пиксели = 255 (непрозрачно), светлые = 0 (прозрачно)
    mask = ((brightness < threshold).astype(np.uint8) * 255)
    return Image.fromarray(mask, 'L')


def load_object_images(objects_dir):
    """
    Загружает все изображения объектов из директории.

    Args:
        objects_dir: путь к директории с объектами

    Returns:
        Список кортежей (PIL Image, PIL Image mask)
    """
    objects = []
    objects_path = Path(objects_dir)

    for img_file in sorted(objects_path.glob("*.jpg")):
        img = Image.open(img_file).convert('RGB')
        mask = create_transparency_mask(img, threshold=240)
        objects.append((img, mask))

    print(f"Загружено {len(objects)} объектов с масками из {objects_dir}")
    return objects


def load_background_images(backgrounds_dir):
    """
    Загружает все фоновые изображения (предобработанные схемы).

    Args:
        backgrounds_dir: путь к директории с фонами

    Returns:
        Список кортежей (PIL Image, имя_файла)
    """
    backgrounds = []
    backgrounds_path = Path(backgrounds_dir)

    for img_file in sorted(backgrounds_path.glob("*.jpg")):
        img = Image.open(img_file).convert('RGB')
        backgrounds.append((img, img_file.stem))

    for img_file in sorted(backgrounds_path.glob("*.png")):
        img = Image.open(img_file).convert('RGB')
        backgrounds.append((img, img_file.stem))

    print(f"Загружено {len(backgrounds)} фоновых изображений из {backgrounds_dir}")
    return backgrounds


def generate_image_with_lines(background_img, objects, num_lines_range=(1, 3),
                              objects_per_line_range=(3, 8), scale_range=(0.8, 1.2),
                              spacing_range=(10, 40), grouping_probability=0.5,
                              group_size_range=(2, 3), group_spacing_multiplier=(2, 4)):
    """
    Генерирует изображение, размещая объекты в линии с возможной группировкой.

    Объекты размещаются в горизонтальные линии, как компоненты на схеме.
    Могут быть равномерно распределены или сгруппированы (группы с маленьким
    промежутком внутри и большим между группами).

    Args:
        background_img: PIL Image фона
        objects: список PIL Images объектов
        num_lines_range: диапазон количества линий (1-3)
        objects_per_line_range: количество объектов в линии (3-8)
        scale_range: диапазон масштабирования объектов
        spacing_range: базовое расстояние между объектами в пикселях
        grouping_probability: вероятность группировки объектов (0-1)
        group_size_range: размер группы (2-3 объекта)
        group_spacing_multiplier: множитель расстояния между группами (2-4x)

    Returns:
        Tuple (PIL Image, list of annotations)
    """
    canvas = background_img.copy()
    img_width, img_height = canvas.size
    annotations = []

    # Определяем количество линий
    num_lines = random.randint(num_lines_range[0], num_lines_range[1])

    # Вычисляем доступную высоту для линий
    line_height_space = img_height // (num_lines + 1)

    for line_idx in range(num_lines):
        # Y-координата линии (распределяем линии равномерно по высоте)
        line_y = line_height_space * (line_idx + 1)

        # Количество объектов в этой линии
        num_objects = random.randint(objects_per_line_range[0], objects_per_line_range[1])

        # Решаем, будет ли группировка в этой линии
        use_grouping = random.random() < grouping_probability

        # Масштабируем объекты и получаем их размеры
        scaled_objects = []
        for _ in range(num_objects):
            obj_img, obj_mask = random.choice(objects)
            scale = random.uniform(scale_range[0], scale_range[1])
            new_width = int(obj_img.width * scale)
            new_height = int(obj_img.height * scale)

            if new_width < 5 or new_height < 5:
                continue

            obj_scaled = obj_img.resize((new_width, new_height), Image.LANCZOS)
            mask_scaled = obj_mask.resize((new_width, new_height), Image.LANCZOS)
            scaled_objects.append((obj_scaled, mask_scaled, new_width, new_height))

        if len(scaled_objects) == 0:
            continue

        # Вычисляем общую ширину линии
        total_obj_width = sum(w for _, _, w, _ in scaled_objects)

        if use_grouping:
            # Группировка: разбиваем на группы
            group_size = random.randint(group_size_range[0], group_size_range[1])
            groups = []
            current_group = []

            for obj_data in scaled_objects:
                current_group.append(obj_data)
                if len(current_group) >= group_size:
                    groups.append(current_group)
                    current_group = []

            if current_group:  # Добавляем последнюю неполную группу
                groups.append(current_group)

            # Расстояния
            base_spacing = random.randint(spacing_range[0], spacing_range[1])
            group_spacing = base_spacing * random.uniform(group_spacing_multiplier[0],
                                                          group_spacing_multiplier[1])

            # Вычисляем общую ширину с учетом группировки
            total_width_with_spacing = (total_obj_width +
                                       base_spacing * (len(scaled_objects) - len(groups)) +
                                       group_spacing * (len(groups) - 1))

            # Начальная X позиция (случайная вместо центрирования)
            if total_width_with_spacing < img_width:
                max_start_x = img_width - total_width_with_spacing
                start_x = random.uniform(0, max_start_x)
            else:
                start_x = 10  # Отступ от края
                # Уменьшаем расстояния если не помещается
                scale_factor = (img_width - 20) / total_width_with_spacing
                base_spacing *= scale_factor
                group_spacing *= scale_factor

            # Размещаем группы
            current_x = start_x
            for group_idx, group in enumerate(groups):
                for obj_idx, (obj_scaled, mask_scaled, obj_width, obj_height) in enumerate(group):
                    # Y позиция с небольшой случайной вариацией
                    y_variation = random.randint(-5, 5)
                    obj_y = line_y - obj_height // 2 + y_variation

                    # Проверяем границы
                    obj_y = max(0, min(obj_y, img_height - obj_height))
                    obj_x = int(current_x)

                    if obj_x + obj_width <= img_width:
                        # Вставляем объект с маской (удаление белого фона)
                        canvas.paste(obj_scaled, (obj_x, obj_y), mask=mask_scaled)

                        # Добавляем аннотацию
                        x_center = (obj_x + obj_width / 2) / img_width
                        y_center = (obj_y + obj_height / 2) / img_height
                        width = obj_width / img_width
                        height = obj_height / img_height
                        annotations.append((0, x_center, y_center, width, height))

                    # Смещаемся на ширину объекта + расстояние внутри группы
                    current_x += obj_width + base_spacing

                # Добавляем расстояние между группами (убираем лишний base_spacing)
                if group_idx < len(groups) - 1:
                    current_x += (group_spacing - base_spacing)

        else:
            # Равномерное распределение без группировки
            spacing = random.randint(spacing_range[0], spacing_range[1])
            total_width_with_spacing = total_obj_width + spacing * (len(scaled_objects) - 1)

            # Начальная X позиция (случайная вместо центрирования)
            if total_width_with_spacing < img_width:
                max_start_x = img_width - total_width_with_spacing
                start_x = random.uniform(0, max_start_x)
            else:
                start_x = 10
                # Уменьшаем расстояния если не помещается
                spacing = int((img_width - 20 - total_obj_width) / max(1, len(scaled_objects) - 1))

            # Размещаем объекты
            current_x = start_x
            for obj_scaled, mask_scaled, obj_width, obj_height in scaled_objects:
                # Y позиция с небольшой случайной вариацией
                y_variation = random.randint(-5, 5)
                obj_y = line_y - obj_height // 2 + y_variation

                # Проверяем границы
                obj_y = max(0, min(obj_y, img_height - obj_height))
                obj_x = int(current_x)

                if obj_x + obj_width <= img_width:
                    # Вставляем объект с маской (удаление белого фона)
                    canvas.paste(obj_scaled, (obj_x, obj_y), mask=mask_scaled)

                    # Добавляем аннотацию
                    x_center = (obj_x + obj_width / 2) / img_width
                    y_center = (obj_y + obj_height / 2) / img_height
                    width = obj_width / img_width
                    height = obj_height / img_height
                    annotations.append((0, x_center, y_center, width, height))

                # Смещаемся на ширину объекта + расстояние
                current_x += obj_width + spacing

    return canvas, annotations


def generate_image_on_background(background_img, objects, num_objects_range=(5, 15),
                                 scale_range=(0.8, 1.2)):
    """
    DEPRECATED: Старая функция со случайным размещением.
    Оставлена для обратной совместимости.
    """
    # Создаем копию фона
    canvas = background_img.copy()
    img_width, img_height = canvas.size

    # Случайное количество объектов
    num_objects = random.randint(num_objects_range[0], num_objects_range[1])

    annotations = []

    for _ in range(num_objects):
        # Случайный выбор объекта
        obj_img, obj_mask = random.choice(objects)

        # Случайный масштаб
        scale = random.uniform(scale_range[0], scale_range[1])
        new_width = int(obj_img.width * scale)
        new_height = int(obj_img.height * scale)

        # Проверка минимального размера
        if new_width < 5 or new_height < 5:
            continue

        # Изменение размера объекта и маски
        obj_scaled = obj_img.resize((new_width, new_height), Image.LANCZOS)
        mask_scaled = obj_mask.resize((new_width, new_height), Image.LANCZOS)

        # Случайная позиция (объект должен быть полностью внутри изображения)
        max_x = img_width - new_width
        max_y = img_height - new_height

        if max_x <= 0 or max_y <= 0:
            continue  # Пропускаем, если объект слишком большой

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Вставляем объект с маской (удаление белого фона)
        canvas.paste(obj_scaled, (x, y), mask=mask_scaled)

        # Вычисляем YOLO аннотацию (нормализованные координаты)
        x_center = (x + new_width / 2) / img_width
        y_center = (y + new_height / 2) / img_height
        width = new_width / img_width
        height = new_height / img_height

        # Class ID = 0 (один класс)
        annotations.append((0, x_center, y_center, width, height))

    return canvas, annotations


def save_yolo_annotation(annotations, output_path):
    """
    Сохраняет аннотации в YOLO формате.

    Args:
        annotations: список (class_id, x_center, y_center, width, height)
        output_path: путь для сохранения .txt файла
    """
    with open(output_path, 'w') as f:
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def generate_dataset_on_real_backgrounds(backgrounds_dir, objects_dir, output_dir,
                                         images_per_combo=100, num_objects_range=(5, 15),
                                         scale_range=(0.8, 1.2), grain_levels=[0, 5, 10, 15]):
    """
    Генерирует YOLO датасет на основе реальных фонов.

    Args:
        backgrounds_dir: директория с предобработанными фонами
        objects_dir: директория с объектами
        output_dir: выходная директория
        images_per_combo: количество изображений на комбинацию (фон + grain)
        num_objects_range: диапазон объектов на изображение
        scale_range: диапазон масштабов объектов
        grain_levels: уровни интенсивности grain
    """
    # Загружаем объекты и фоны
    objects = load_object_images(objects_dir)
    backgrounds = load_background_images(backgrounds_dir)

    if len(objects) == 0:
        print("ОШИБКА: Не найдено объектов!")
        return

    if len(backgrounds) == 0:
        print("ОШИБКА: Не найдено фоновых изображений!")
        return

    # Создаем выходные директории
    output_path = Path(output_dir)
    (output_path / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'train' / 'labels').mkdir(parents=True, exist_ok=True)

    # Подсчет общего количества изображений
    total_images = len(backgrounds) * len(grain_levels) * images_per_combo

    print(f"\n{'='*60}")
    print(f"ГЕНЕРАЦИЯ ДАТАСЕТА С ЛИНЕЙНЫМ РАЗМЕЩЕНИЕМ")
    print(f"{'='*60}")
    print(f"Фонов: {len(backgrounds)}")
    print(f"Объектов: {len(objects)}")
    print(f"Уровней grain: {len(grain_levels)} {grain_levels}")
    print(f"Изображений на комбинацию: {images_per_combo}")
    print(f"Режим размещения: Линии с группировкой")
    print(f"  - Линий на изображение: 1-3")
    print(f"  - Объектов в линии: 2-17")
    print(f"  - Вероятность группировки: 50%")
    print(f"  - Равномерное расстояние: 15-50 пикс")
    print(f"  - Расстояние между группами: 2-20x базового")
    print(f"  - Позиция ряда: случайная (без центрирования)")
    print(f"  - Масштаб объектов: {scale_range[0]}x-{scale_range[1]}x")
    print(f"{'='*60}")
    print(f"ВСЕГО ИЗОБРАЖЕНИЙ: {total_images}")
    print(f"{'='*60}\n")

    # Проверяем существующие изображения
    existing_images = list((output_path / 'train' / 'images').glob('*.jpg'))
    num_existing = len(existing_images)

    if num_existing > 0:
        print(f"Найдено {num_existing} существующих изображений")
        print(f"Будут сгенерированы только недостающие\n")

    global_idx = 0
    generated_count = 0

    # Генерация
    for bg_img, bg_name in backgrounds:
        for grain_intensity in grain_levels:
            print(f"\nФон: {bg_name} | Grain: {grain_intensity}")
            print(f"-" * 60)

            for i in range(images_per_combo):
                # Формируем имя файла
                img_filename = f"train_{global_idx:05d}.jpg"
                img_path = output_path / 'train' / 'images' / img_filename
                ann_path = output_path / 'train' / 'labels' / f"train_{global_idx:05d}.txt"

                # Пропускаем если файл уже существует
                if img_path.exists() and ann_path.exists():
                    global_idx += 1
                    continue

                # Генерируем изображение с объектами в линиях
                img, annotations = generate_image_with_lines(
                    background_img=bg_img,
                    objects=objects,
                    num_lines_range=(1, 3),
                    objects_per_line_range=(2, 17),
                    scale_range=scale_range,
                    spacing_range=(15, 50),
                    grouping_probability=0.5,
                    group_size_range=(2, 3),
                    group_spacing_multiplier=(2, 20)
                )

                num_objs = len(annotations)

                # Применяем grain
                img = apply_grain(img, grain_intensity)

                # Конвертируем в grayscale
                img = img.convert('L')

                # Сохраняем
                img.save(img_path, quality=95)
                save_yolo_annotation(annotations, ann_path)

                generated_count += 1
                global_idx += 1

                # Лог каждые 10 изображений
                if (i + 1) % 10 == 0 or (i + 1) <= 5:
                    progress_pct = ((i + 1) / images_per_combo) * 100
                    print(f"  [{progress_pct:5.1f}%] {i + 1}/{images_per_combo} | "
                          f"Объектов: {num_objs:2d}")

            print(f"[OK] Комбинация завершена")

    print(f"\n{'='*60}")
    print(f"ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
    print(f"{'='*60}")
    print(f"Сгенерировано новых изображений: {generated_count}")
    print(f"Всего изображений в train: {global_idx}")
    print(f"Выходная директория: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Конфигурация
    BACKGROUNDS_DIR = "results/preprocessed_20251018_191740"
    OBJECTS_DIR = "data/datasets/objects-v2"
    OUTPUT_DIR = "data/yolo_dataset"

    # Параметры генерации (Тестовый режим с линейным размещением)
    IMAGES_PER_COMBO = 10  # На каждую комбинацию фон+grain (для тестирования)
    NUM_OBJECTS_RANGE = (5, 15)  # Не используется с новым алгоритмом
    SCALE_RANGE = (0.8, 1.2)
    GRAIN_LEVELS = [0, 5, 10, 15]  # 4 уровня grain

    # Итого: 3 фона × 4 grain × 10 = 120 изображений (тестовая генерация)

    print("="*60)
    print("YOLO DATASET GENERATOR - ЛИНЕЙНОЕ РАЗМЕЩЕНИЕ")
    print("="*60)
    print(f"Фоны: {BACKGROUNDS_DIR}")
    print(f"Объекты: {OBJECTS_DIR}")
    print(f"Выход: {OUTPUT_DIR}")
    print(f"Режим: Линии с группировкой (тестовый, 10 изображений/комбо)")
    print("="*60)

    # Генерация
    generate_dataset_on_real_backgrounds(
        backgrounds_dir=BACKGROUNDS_DIR,
        objects_dir=OBJECTS_DIR,
        output_dir=OUTPUT_DIR,
        images_per_combo=IMAGES_PER_COMBO,
        num_objects_range=NUM_OBJECTS_RANGE,
        scale_range=SCALE_RANGE,
        grain_levels=GRAIN_LEVELS
    )
