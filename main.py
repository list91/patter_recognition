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
    _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

    # Используем пороговое изображение как маску для выделения объекта
    masked_img = np.zeros_like(img1)
    masked_img[thresh != 0] = img1[thresh != 0]

    # Фон делаем белым
    masked_img[thresh == 0] = 255

    # Отображение или сохранение результата
    cv2.imshow('Detected Object', masked_img)
    # cv2.imshow('Detected Object', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main("1","clear")
main("2","3")
main("4","2")
