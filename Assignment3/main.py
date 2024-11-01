import numpy as np
from torchvision.datasets import MNIST
import time

LR = 0.02
EPOCHS = 20
BATCH_SIZE = 100
HIDDEN_NODES = 100
INPUT_DIM = 784
OUTPUT_SIZE = 10
NUM_CLASSES = 10  # Digits 0-9
DROPOUT_PROB = 0.8


def download_mnist(is_train: bool):
    dataset = MNIST(
        root="./data",
        transform=lambda x: np.array(x).flatten(),
        download=True,
        train=is_train,
    )

    mnist_data = []
    mnist_labels = []

    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append([int(item == label) for item in range(NUM_CLASSES)])  # One-hot encoding

    return mnist_data, mnist_labels


# Define activation and derivative functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -100, 100)))  # Vectorized and clipped input


def sigmoid_derive(x):
    value = sigmoid(x)
    return value * (1 - value)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Improve stability with softmax
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class Network:
    def __init__(self, input_dim=INPUT_DIM, hidden_nodes=HIDDEN_NODES, output_size=OUTPUT_SIZE, learning_rate=LR):
        self.a2 = None
        self.z2 = None
        self.a1 = None
        self.z1 = None
        self.learning_rate = learning_rate
        self.w1 = np.random.randn(input_dim, hidden_nodes)
        self.b1 = np.random.randn(hidden_nodes)
        self.w2 = np.random.randn(hidden_nodes, output_size)
        self.b2 = np.random.randn(output_size)

    def forward(self, x):
        # Input to hidden layer
        self.z1 = x.dot(self.w1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1.dot(self.w2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backpropagation(self, x, y, p_on):
        # Randomly deactivate hidden nodes
        active_hidden = np.random.choice([0, 1], size=(self.w1.shape[1],), p=[1 - p_on, p_on])
        active_hidden = np.tile(active_hidden, (x.shape[0], 1))

        # Forward pass
        self.z1 = np.dot(x, self.w1) + self.b1 # Linear combination for hidden layer
        self.a1 = sigmoid(self.z1) # Activation function (sigmoid) for hidden layer
        self.a1 *= active_hidden  # Apply dropout by zeroing out inactive nodes

        self.z2 = np.dot(self.a1, self.w2) + self.b2 # Linear combination for output layer
        self.a2 = softmax(self.z2) # Activation function (softmax) for output layer

        # Backward pass for output layer
        delta2 = y - self.a2 # Error at output layer (difference between true and predicted)
        grad_w2 = np.dot(self.a1.T, delta2) # Gradient of the weights for output layer
        grad_b2 = np.sum(delta2, axis=0) # Gradient of the biases for output layer

        # Backward pass for hidden layer
        delta1 = np.dot(delta2, self.w2.T) * sigmoid_derive(self.z1) # Error at hidden layer
        delta1 *= active_hidden  # Apply dropout to the gradient by zeroing out inactive nodes
        grad_w1 = np.dot(x.T, delta1) # Gradient of the weights for hidden layer
        grad_b1 = np.sum(delta1, axis=0) # Gradient of the biases for hidden layer

        self.w1 += self.learning_rate * grad_w1 # Update weights for hidden layer
        self.b1 += self.learning_rate * grad_b1 # Update biases for hidden layer
        self.w2 += self.learning_rate * grad_w2 # Update weights for output layer
        self.b2 += self.learning_rate * grad_b2 # Update biases for output layer

    def train(self, x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, p_on=DROPOUT_PROB):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            # Shuffle data at each epoch
            permutation = np.random.permutation(x.shape[0])
            x_shuffled, y_shuffled = x[permutation], y[permutation]

            for i in range(0, x.shape[0], batch_size):
                batch_x = x_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                self.backpropagation(batch_x, batch_y, p_on)

    def accuracy(self, x, y):
        predictions = self.forward(x)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predicted_labels == true_labels)

if __name__ == '__main__':
    training_data, training_labels = download_mnist(True)
    testing_data, testing_labels = download_mnist(False)

    # Normalize data
    training_data = np.array(training_data, dtype=np.float64) / 255
    training_labels = np.array(training_labels)
    testing_data = np.array(testing_data, dtype=np.float64) / 255
    testing_labels = np.array(testing_labels)

    model = Network(learning_rate=LR)
    time_start = time.time()
    model.train(training_data, training_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, p_on=DROPOUT_PROB)
    time_end = time.time()
    print(f"Elapsed time: {(time_end - time_start) / 60} minutes")
    print(f"Final Training Accuracy: {model.accuracy(training_data, training_labels) * 100}%")
    print(f"Final Validation Accuracy: {model.accuracy(testing_data, testing_labels) * 100}%")

# Results
# Epoch = 20, LR = 0.02 => T: 97.93%  V: 96.39
# Epoch = 50, LR = 0.02 => T: 99.29%  V: 96.98
# Epoch = 200, LR = 0.02 => T: 99.99%  V: 97.15