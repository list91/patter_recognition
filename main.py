import cv2
import numpy as np

def main(i, j):
    img1 = cv2.imread(f'src\\detect_datasets\\1\\{i}.jpg', 0)
    img2 = cv2.imread(f'src\\detect_datasets\\1\\{j}.jpg', 0)

    # Применение медианного фильтра
    # img1 = cv2.medianBlur(img1, 5)
    # img2 = cv2.medianBlur(img2, 5)

    # Вычисление разности
    diff = cv2.absdiff(img1, img2)

    # Пороговая обработка для выделения изменений
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Находим контуры на пороговом изображении
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ищем контур с наибольшей площадью
    max_contour = max(contours, key=cv2.contourArea)
    
    # Получаем координаты ограничивающего прямоугольника
    x, y, w, h = cv2.boundingRect(max_contour)

    # Рисуем зелёный прямоугольник на исходном изображении img1
    result_img = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображаем результат
    cv2.imshow('Detected Object', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main("3", "clear")
main("4", "clear")
# main("1", "clear")
# main("2", "3")
main("4", "2")
