import cv2
import numpy as np
import img

class Perceptron:
    def __init__(self, input_size, hidden_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, 2) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, 2))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, input_data, target_output, learning_rate, epochs):
        # print(i)
        for epoch in range(epochs):
            # Forward pass
            hidden_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
            hidden_output = self.sigmoid(hidden_input)
            final_output = self.sigmoid(np.dot(hidden_output, self.weights_hidden_output) + self.bias_output)

            # Calculate error
            error = target_output - final_output
            output_delta = error * self.sigmoid_derivative(final_output)
            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

            # Update weights and biases
            self.weights_hidden_output += learning_rate * hidden_output.T.dot(output_delta)
            self.weights_input_hidden += learning_rate * input_data.T.dot(hidden_delta)
            self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
            self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

    def predict(self, input_data):
        hidden_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        final_output = self.sigmoid(np.dot(hidden_output, self.weights_hidden_output) + self.bias_output)
        return final_output


input_size = 22 * 36 * 3  # Size of the input layer, 22x36 image, 3 channels (RGB)
hidden_size = 1000  # Size of the hidden layer

perceptron = Perceptron(input_size, hidden_size)

file_list = img.get_file_list("datasets")
for i in file_list:
    image = cv2.imread(i)
    if image is None:
        print("Failed to load image:", i)
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb / 255.0
    image_flattened = image_normalized.flatten().reshape(1, -1)  # Flatten the image to fit the perceptron input

    truth_value = np.array([[0.9, 0.1]])  # Example truth values: 90% "yes", 10% "no"

    perceptron.train(image_flattened, truth_value, learning_rate=0.01, epochs=10)

test_image = cv2.imread("tests/dataset1.png")
# test_image = cv2.imread(file_list[2])
test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image_normalized = test_image_rgb / 255.0
test_image_flattened = test_image_normalized.flatten().reshape(1, -1)

prediction = perceptron.predict(test_image_flattened)
print("Prediction:", prediction)
