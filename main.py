import cv2

def main(i):
    img1 = cv2.imread(f'src\\detect_datasets\\1\\{str(i)}.jpg', 0)
    img2 = cv2.imread(f'src\\detect_datasets\\1\\clear.jpg', 0)

    # Применение медианного фильтра
    img1 = cv2.medianBlur(img1, 5)
    img2 = cv2.medianBlur(img2, 5)

    # Вычисление разности
    diff = cv2.absdiff(img1, img2)

    # Пороговая обработка для выделения изменений
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Отображение или сохранение результата
    cv2.imshow('Difference', diff)
    cv2.imshow('Thresholded', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main(1)
main(2)
main(3)
main(4)