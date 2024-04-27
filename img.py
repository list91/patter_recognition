import os

def get_file_list(directory):
    # Используем метод listdir для получения списка файлов и директорий в указанной директории
    file_list = os.listdir(directory)
    
    # Возвращаем список файлов (без директорий)
    return [directory+"/"+f for f in file_list if os.path.isfile(os.path.join(directory, f))]

import cv2

def load_image_as_grid(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Получение высоты и ширины изображения
    height, width, _ = image.shape
    
    # Преобразование изображения в сетку пикселей
    pixel_grid = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(image[y, x])
        pixel_grid.append(row)
    
    return image, pixel_grid

import cv2
import numpy as np

def match_template_explicit(large_image_path, dataset_directory, threshold=0.8):
    # Загрузка большого изображения
    large_image = cv2.imread(large_image_path)
    
    # Получение списка файлов датасета
    dataset_files = get_file_list(dataset_directory)
    
    results = []
    for file in dataset_files:
        # Загрузка шаблона изображения из датасета
        template = cv2.imread(file)
        template_height, template_width = template.shape[:2]

        # Проверка на корректность загруженных изображений
        if template is None or large_image is None:
            continue
        
        # Инициализация скользящего окна для перебора всех позиций шаблона в большом изображении
        for y in range(large_image.shape[0] - template_height + 1):
            for x in range(large_image.shape[1] - template_width + 1):
                # Вырезаем часть изображения размером с шаблон
                window = large_image[y:y + template_height, x:x + template_width]
                
                res = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
                # print(res)                
                max_val = np.max(res)
                # Если найдено совпадение выше порога, добавляем его в результаты
                if max_val >= threshold:
                    results.append((file, (x, y), max_val))

    return results


# directory_path = "vis/datasets"  # Укажите путь к вашей директории
    
# # Пример использования функции:C:\Users\ssdwq\Downloads\Telegram Desktop\vis\vis\extracted_imagesMarker\2024_04_24_0pp_Kleki.png
# large_image_path = "vis/extracted_imagesMarker/2024_04_24_0pp_Kleki.png"  # Укажите путь к вашему большому изображению
# dataset_directory = directory_path  # Укажите путь к директории с датасетом
# found_matches = match_template_explicit(large_image_path, dataset_directory)
# for i in found_matches:
#     print(i)