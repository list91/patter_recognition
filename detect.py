import cv2
import numpy as np

def main(i, j):
    img1 = cv2.imread(f'src\\detect_datasets\\1\\{i}.jpg', 0)
    img2 = cv2.imread(f'src\\detect_datasets\\1\\{j}.jpg', 0)

    # Вычисление разности
    diff = cv2.absdiff(img1, img2)

    # Пороговая обработка для выделения изменений
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Находим контуры на пороговом изображении
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    masked_img = np.zeros_like(img1)
    masked_img[thresh != 0] = img1[thresh != 0]

    # Фон делаем белым
    masked_img[thresh == 0] = 255

    # Ищем контур с наибольшей площадью
    max_contour = max(contours, key=cv2.contourArea)
    
    # Получаем координаты ограничивающего прямоугольника
    x, y, w, h = cv2.boundingRect(max_contour)

    # Рисуем зелёный прямоугольник на исходном изображении img1
    result_img = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Detected Object', masked_img[y:y+h, x:x+w])
    # cv2.imshow('Detected Object', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_move_matrix_pixels(pic1, pic2):
    img1 = cv2.imread(pic1, 0)
    img2 = cv2.imread(pic2, 0)
    diff = cv2.absdiff(img1, img2)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masked_img = np.zeros_like(img1)
    masked_img[thresh != 0] = img1[thresh != 0]
    masked_img[thresh == 0] = 0
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return masked_img[y:y+h, x:x+w]

