import detect

detect.main("1", "2")
# detect.main("4", "clear")
# detect.main("1", "clear")
# detect.main("2", "3")
# detect.main("4", "2")

import cv2
import numpy as np

def load_grayscale_image_as_np(path):
    try:
        # Загрузка изображения с использованием OpenCV
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {path}")
        
        # Преобразование изображения в NumPy массив
        image_np = np.array(image)
        
        return image_np
    
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return None

import matplotlib.pyplot as plt
def crop_array(array, start_row, end_row, start_col, end_col):
    cropped_array = array[start_row:end_row, start_col:end_col]
    return cropped_array
def show_image_from_array(image_array):
    # Создаем изображение из массива NumPy
    plt.imshow(image_array, cmap='gray')
    plt.axis('off')  # Убираем оси координат
    plt.show()
# Пример использования нейронной сети для извлечения признаков из изображения
def main():
    grayscale_image_np = load_grayscale_image_as_np("src\\detect_datasets\\1\\1.jpg")

    cropped_array = crop_array(grayscale_image_np, 
                               0, 400, 
                               0, 400)
    show_image_from_array(cropped_array)

    print(grayscale_image_np.shape[0])
    print(grayscale_image_np.shape[1])

    feature_map = cropped_array

    # Размер ядра усредняющей свертки 2x2
    kernel_size = 2

    # Размер выходной карты после усредняющей свертки
    output_size = feature_map.shape[0] // kernel_size

    # Создадим массив для хранения результата усредняющей свертки
    smoothed_features = np.zeros((output_size, output_size))

    # Применим усредняющую свертку
    for i in range(output_size):
        for j in range(output_size):
            # Область на карте признаков, на которую применяем усредняющую свертку
            patch = feature_map[i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size]
            # Вычисляем среднее значение
            smoothed_features[i, j] = np.mean(patch)

    # Выводим результаты
    print("Исходная карта признаков:")
    print(feature_map)
    print("\nУменьшенная карта признаков после усредняющей свертки:")
    print(smoothed_features)
    print(smoothed_features.shape[0])
    print(smoothed_features.shape[1])
    show_image_from_array(smoothed_features)

if __name__ == "__main__":
    main()


