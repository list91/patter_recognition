import numpy as np
import detect
# Функция активации - сигмоида
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная функции активации сигмоиды
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Нейронная сеть класса
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Инициализация весов со случайными значениями
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
        
    def forward(self, X):
        # Прямое распространение
        
        # Скрытый слой
        self.hidden_sum = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_activation = sigmoid(self.hidden_sum)
        
        # Выходной слой
        output_sum = np.dot(self.hidden_activation, self.weights_hidden_output) + self.bias_output
        output = sigmoid(output_sum)
        
        return output
    
    def backward(self, X, y, output, learning_rate):
        # Обратное распространение
        
        # Вычисляем градиент для выходного слоя
        error = y - output
        output_delta = error * sigmoid_derivative(output)
        
        # Обновляем веса и смещения для выходного слоя
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_activation.T, output_delta)
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        
        # Вычисляем градиент для скрытого слоя
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_activation)
        
        # Обновляем веса и смещения для скрытого слоя
        self.weights_input_hidden += learning_rate * np.dot(X.T, hidden_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
    
    def train(self, X, y, epochs, learning_rate):
        # Обучение нейронной сети
        for epoch in range(epochs):
            # Прямое распространение
            output = self.forward(X)
            
            # Обратное распространение и обновление параметров
            self.backward(X, y, output, learning_rate)
            
            # Вычисляем и выводим среднюю ошибку на каждой эпохе (для наглядности)
            loss = np.mean(np.abs(y - output))
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    def predict(self, X):
        # Предсказание на основе текущих весов и смещений
        return self.forward(X)

# Пример использования

# Создаем набор данных для обучения (входные данные и соответствующие выходные значения)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Создаем экземпляр нейронной сети
# input_size = X.shape[1]
# hidden_size = 4  # количество нейронов в скрытом слое
# output_size = y.shape[1]

# nn = NeuralNetwork(input_size, hidden_size, output_size)

# # Обучаем нейронную сеть
# nn.train(X, y, epochs=10000, learning_rate=0.1)

# # Делаем предсказание
# predictions = nn.predict(X)
# print("\nFinal predictions:")
# print(predictions)



def apply_convolution(kernel, input_matrix):
    # Размеры входной матрицы
    in_rows, in_cols = input_matrix.shape
    
    # Размеры ядра свертки
    kernel_rows, kernel_cols = kernel.shape
    
    # Размеры выходной матрицы (задаем 50x50)
    out_rows, out_cols = 100, 100
    
    # Шаг свертки (как регулировать шаг, чтобы получить 50x50 матрицу)
    stride_rows = max((in_rows - kernel_rows) // (out_rows - 1), 1)
    stride_cols = max((in_cols - kernel_cols) // (out_cols - 1), 1)
    
    # Применяем свертку с заданным шагом
    result = np.zeros((out_rows, out_cols))
    for i in range(0, out_rows):
        for j in range(0, out_cols):
            # Определяем начальные координаты входной матрицы для текущего шага
            start_row = i * stride_rows
            start_col = j * stride_cols
            # Область входной матрицы для текущего шага
            input_patch = input_matrix[start_row:start_row + kernel_rows, start_col:start_col + kernel_cols]
            # Применяем свертку
            result[i, j] = np.sum(input_patch * kernel)
    
    return result

# Пример использования функции
# Создаем матрицу коэффициентов размером 100x100
# input_matrix = np.random.rand(640, 420)

move_map = detect.get_move_matrix_pixels(
    'src\\detect_datasets\\1\\1.jpg',
    'src\\detect_datasets\\1\\2.jpg'
)
# print(move_map.shape[1])
# Создаем ядро свертки размером 3x3

kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

# Применяем свертку и получаем результат размером 50x50
output_matrix = apply_convolution(kernel, move_map)
# print(move_map)
# print(output_matrix)
# input_size = X.shape[1]
# hidden_size = 4  # количество нейронов в скрытом слое
# output_size = y.shape[1]

# nn = NeuralNetwork(input_size, hidden_size, output_size)

# # Обучаем нейронную сеть
# nn.train(X, y, epochs=10000, learning_rate=0.1)

# # Делаем предсказание
# predictions = nn.predict(X)
# print("\nFinal predictions:")
# print(predictions)
