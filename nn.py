import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Загрузка модели из TensorFlow Hub
model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'
detector = hub.load(model_handle)

# Получение списка доступных сигнатур модели
print("Доступные сигнатуры модели:", detector.signatures.keys())

# Выбор сигнатуры для обнаружения объектов
signature_keys = list(detector.signatures.keys())
selected_signature_key = signature_keys[0]  # Выбираем первую доступную сигнатуру

# Функция для обработки изображения и обнаружения объектов
def detect_people(image_path):
    # Загрузка изображения с диска
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (300, 300))

    # Преобразование в тензор и добавление размерности пакета
    input_tensor = tf.convert_to_tensor(image_resized)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    input_tensor = tf.cast(input_tensor, dtype=tf.uint8)

    # Обнаружение объектов на изображении
    result = detector.signatures[selected_signature_key](input_tensor=input_tensor)

    # Интересующие нас классы и их индексы
    interesting_classes = ['background', 'person']  # Добавлен 'background' как 0-й класс
    class_ids = result['detection_classes'][0].numpy()
    scores = result['detection_scores'][0].numpy()
    boxes = result['detection_boxes'][0].numpy()

    # Отрисовка результатов на изображении
    image_with_boxes = image_resized.copy()
    for i in range(len(class_ids)):
        class_id = int(class_ids[i])
        score = scores[i]
        box = boxes[i]

        if score >= 0.5 and class_id < len(interesting_classes):
            if interesting_classes[class_id] == 'person':
                ymin, xmin, ymax, xmax = box
                left, right, top, bottom = int(xmin * 300), int(xmax * 300), int(ymin * 300), int(ymax * 300)
                cv2.rectangle(image_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)

    return image_with_boxes

# Пример использования
image_path = 'src\\detect_datasets\\1\\5.jpg'  # Замените на реальный путь к вашему изображению
image_with_boxes = detect_people(image_path)

# Отображение результата
cv2.imshow('Detected People', image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
