from pdf2image import convert_from_path
import PyPDF2
from PIL import Image
import cv2
import numpy as np
import os
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def main():
    class SimpleNeuralNetwork:
        def __init__(self, input_size, hidden_size, output_size):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
            self.b1 = np.zeros((1, self.hidden_size))
            self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
            self.b2 = np.zeros((1, self.output_size))

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_derivative(self, x):
            return x * (1 - x)
        
        def train(self, X, y, epochs, learning_rate):
            for epoch in range(epochs):
                hidden_layer_input = np.dot(X, self.W1) + self.b1
                hidden_layer_output = self.sigmoid(hidden_layer_input)
                final_layer_input = np.dot(hidden_layer_output, self.W2) + self.b2
                final_layer_output = self.sigmoid(final_layer_input)
                error = y - final_layer_output
                d_output = error * self.sigmoid_derivative(final_layer_output)
                error_hidden_layer = d_output.dot(self.W2.T)
                d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)
                self.W2 += hidden_layer_output.T.dot(d_output) * learning_rate
                self.b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
                self.W1 += X.T.dot(d_hidden_layer) * learning_rate
                self.b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

        def predict(self, X):
            hidden_layer_input = np.dot(X, self.W1) + self.b1
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            final_layer_input = np.dot(hidden_layer_output, self.W2) + self.b2
            final_layer_output = self.sigmoid(final_layer_input)
            return final_layer_output * 100

    def get_file_list(directory):
        file_list = os.listdir(directory)
        return [directory + "/" + f for f in file_list if os.path.isfile(os.path.join(directory, f))]

    def load_image_as_grayscale(image_path):
        # print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        normalized_image = image / 255.0
        return normalized_image.reshape(1, -1)  # Reshape the image to 1D

    # Загрузка и подготовка данных
    directory_path = "vis/datasets"
    file_list = get_file_list(directory_path)
    images = np.array([load_image_as_grayscale(img_path) for img_path in file_list])
    images = images.reshape(images.shape[0], -1)  # Убедитесь, что данные правильно сформированы

    # Создание искусственных меток (для примера)
    labels = np.random.randint(0, 2, (images.shape[0], 1))  # Примерные метки: 0 или 1

    # Инициализация и обучение сети
    nn = SimpleNeuralNetwork(input_size=22*36, hidden_size=100, output_size=1)
    nn.train(images, labels, epochs=100, learning_rate=0.1)

    # Предсказание для нового изображения
    new_image = load_image_as_grayscale("vis/tests/dataset1.png")  # Путь к новому изображению
    # new_image = load_image_as_grayscale("vis/datasets/dataset1.png")  # Путь к новому изображению
    similarity_percentage = nn.predict(new_image)
    print(similarity_percentage[0][0])
    return similarity_percentage[0][0]
    # print(f"Процент сходства: {similarity_percentage[0][0]:.2f}%")
# min = 100
# max = 0
# for _ in range(39):
#     q = main()
#     if q>max:
#         max = q
#     if q<min:
#         min = q
# print("\n\n"+str(min)+" - min \n"+str(max)+" - MAX")
# print("#"*25)

# import numpy as nps
# import q
class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def train(self, input_data, target_output, learning_rate, epochs):
        for epoch in range(epochs):
            hidden_input = np.dot(input_data, self.weights_input_hidden)
            hidden_output = self.sigmoid(hidden_input)
            output = np.dot(hidden_output, self.weights_hidden_output)
            output_final = self.softmax(output)

            error = target_output - output_final
            output_delta = error

            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

            self.weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
            self.weights_input_hidden += input_data.T.dot(hidden_delta) * learning_rate

    def predict(self, input_data):
        hidden_input = np.dot(input_data, self.weights_input_hidden)
        hidden_output = self.sigmoid(hidden_input)
        output = np.dot(hidden_output, self.weights_hidden_output)
        return self.softmax(output), output
# directory_path = "vis/datasets"  # Укажите путь к вашей директории
# file_list = q.get_file_list(directory_path)
# for i in file_list:
#     image, pixel_grid = q.load_image_as_grid(i)
#     print(pixel_grid)

# perceptron = Perceptron(input_size=2, hidden_size=4, output_size=2)
# input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# target_output = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
# perceptron.train(input_data, target_output, learning_rate=0.1, epochs=1000)
# predictions, q = perceptron.predict(input_data)
# print(predictions, q)


def find_color_pixels(image_path, target_color, deviation=15):
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Перевод изображения в формат RGB (OpenCV читает изображения в формате BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Определение размеров изображения
    height, width, _ = image_rgb.shape
    
    # Преобразование целевого цвета в формат numpy array
    target_color = np.array(target_color)
    
    # Инициализация списка для хранения координат найденных пикселей
    color_pixels = []
    
    # Проход по всем пикселям изображения
    for y in range(height):
        for x in range(width):
            # Получение цвета пикселя
            pixel_color = image_rgb[y, x]
            
            # Проверка, соответствует ли цвет пикселя целевому цвету с учетом отклонений
            if np.all(np.abs(pixel_color - target_color) <= deviation):
                # Добавление координат пикселя в список найденных пикселей
                color_pixels.append((x, y))
    
    return color_pixels

def save_image_fragments(image_path, color_pixels, output_folder):
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Создание папки для сохранения выделенных фрагментов, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Перебор всех найденных пикселей и сохранение выделенных фрагментов
    for i, (x, y) in enumerate(color_pixels):
        fragment = image[y - 4:y + 32, x - 4:x + 18]  # Вырезаем фрагмент изображения
        fragment_path = os.path.join(output_folder, f'dataset{i}.png')  # Путь для сохранения фрагмента
        cv2.imwrite(fragment_path, fragment)  # Сохраняем фрагмент

# Пример использования функций
# image_path = 'extracted_imagesMarker/2024_04_24_0pp_Kleki.png'  # Путь к вашему изображению
# target_color = (8, 241, 11)      # Целевой цвет в формате RGB
# deviation = 30  # Отклонение для каждого канала RGB

# color_pixels = find_color_pixels(image_path, target_color, deviation)
# print("Координаты найденных пикселей с цветом", target_color, "и отклонением", deviation, ":", color_pixels)

# # Путь для сохранения выделенных фрагментов
# output_folder = 'datasets'

# # Сохранение выделенных фрагментов с нумерацией в названиях файлов
# save_image_fragments(image_path, color_pixels, output_folder)

def extract_images_from_pdf(pdf_path, output_folder):
    pdf_reader = PyPDF2.PdfFileReader(pdf_path)
    
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        xObject = page['/Resources']['/XObject'].getObject()
        
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                data = xObject[obj].getData()
                image = Image.frombytes("RGB", size, data)
                image.save(f"{output_folder}/page{page_num+1}_{obj[1:]}.jpg")
def extract_images_from_pdf(pdf_path, output_folder):
    # Извлечь изображения из PDF
    images = convert_from_path(pdf_path)

    # Сохранить изображения
    for i, image in enumerate(images):
        image_path = f"{output_folder}/page{i+1}.jpg"
        image.save(image_path, 'JPEG')

# # Путь к PDF файлу и папка для сохранения изображений
# pdf_path = "pdfs/012-2015-2, 3-ЭО.pdf"
# output_folder = "extracted_images"

# # Вызов функции для извлечения изображений
# extract_images_from_pdf(pdf_path, output_folder)
