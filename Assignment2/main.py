import numpy as np
from torchvision.datasets import MNIST

LR = 0.001
EPOCHS = 100
BATCH_SIZE = 100
INPUT_DIM = 784
OUTPUT_SIZE = 10
NUM_CLASSES = 10  # Digits 0-9


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
        mnist_labels.append([int(item == label) for item in range(NUM_CLASSES)]) # One-hot encoding

    return mnist_data, mnist_labels


def softmax(weighted_sum):
    w_pow = np.power(np.e, weighted_sum, dtype=np.float64)
    return np.divide(w_pow, w_pow.sum())


# def cross_entropy(prob, target):
#     return np.negative(target).dot(np.log2(prob, dtype=np.float64))


# training function that updates weights and bias using gradient descent
def update_weights(inputs, target_labels, training_weights, training_bias):
    # Compute raw scores (weighted sums)
    raw_scores = inputs.dot(training_weights) + training_bias
    predicted_probs = np.array([softmax(score) for score in raw_scores])
    error_signal = LR * (target_labels - predicted_probs)

    return inputs.T.dot(error_signal), error_signal.sum(axis=0)


# testing
def evaluate_model(test_data, test_labels, test_weights, test_bias):
    correct_predictions = 0

    for data_point, actual_label in zip(test_data, test_labels):
        predicted_probs = softmax(data_point.dot(test_weights) + test_bias)
        if actual_label[np.argmax(predicted_probs)] == 1:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct_predictions} / {len(test_data)} correct)")
    return accuracy


if __name__ == '__main__':
    training_data, training_labels = download_mnist(True)
    testing_data, testing_labels = download_mnist(False)

    # normalize data
    training_data = np.array(training_data, dtype=np.float64) / 255
    testing_data = np.array(testing_data, dtype=np.float64) / 255

    # initialize weights and bias
    weights = np.zeros((INPUT_DIM, NUM_CLASSES), dtype=np.float64)
    bias = np.zeros(NUM_CLASSES, dtype=np.float64)

    dataset_size = len(training_data)

    # training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1} / {EPOCHS}")

        # shuffle training data at the start of each epoch
        shuffled_data = list(zip(training_data, training_labels))
        np.random.shuffle(shuffled_data)
        training_data, training_labels = zip(*shuffled_data)

        for batch_data, batch_labels in zip(
                np.array_split(training_data, dataset_size // BATCH_SIZE),
                np.array_split(training_labels, dataset_size // BATCH_SIZE),
        ):
            weight_updates, bias_updates = update_weights(batch_data, batch_labels, weights, bias)

            weights += weight_updates
            bias += bias_updates

    # avg accuracy is 92% for LR = 0.001 and EPOCHS = 100 (92.61%)
    evaluate_model(testing_data, testing_labels, weights, bias)