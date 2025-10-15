from ultralytics.utils.plotting import Annotator
import cv2

img = cv2.imread("data/images/train/train_01.jpg")
h, w = img.shape[:2]
annotator = Annotator(img)

print(f"Изображение: {img.shape}") # Выводим размеры изображения

with open("data/labels/train/train_01.txt") as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.strip().split())
        print(f"Прочитано: класс={cls}, x={x}, y={y}, w={bw}, h={bh}") # Отладочный вывод
        # Денормализация координат
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        print(f"Денормализовано: ({x1}, {y1}) -> ({x2}, {y2})") # Отладочный вывод
        annotator.box_label([x1, y1, x2, y2], label=f"Class {int(cls)}", color=(0, 255, 0))

cv2.imshow("Ground Truth", img)
cv2.waitKey(0)
cv2.destroyAllWindows()