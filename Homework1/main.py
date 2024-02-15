import pickle, gzip
import numpy as np
import matplotlib.pyplot as plt


def get_data():
    with gzip.open("mnist.pkl.gz", "rb") as fd:
        tr1, val, tr2 = pickle.load(fd, encoding="latin")
    tr1_in, tr1_label = tr1
    val_in, val_label = val
    return tr1_in, tr1_label, val_in, val_label


class MyPerceptron:
    def __init__(self, digit):
        self.digit = digit
        self.w = 1 - np.random.rand(784) * 2
        self.beta = 1 - np.random.rand() * 2

    def train(self, alpha, label, input):
        self.w += input * alpha * label
        self.beta += label * alpha

    def add_batch(self, w, beta):
        self.w += w
        self.beta += beta

    def test(self, instance):
        return self.w.dot(instance) + self.beta


def perceptron_online_training(alpha, iterations, train_input, train_label):
    my_perceptrons = []
    for digit in range(10):
        print(f"{digit} - online training")
        perceptron = MyPerceptron(digit)
        for _ in range(iterations):
            for i in np.arange(train_label.size):
                if train_label[i] == digit:
                    label = np.float64(1)
                else:
                    label = np.float64(-1)

                f = label * perceptron.test(train_input[i])

                if f <= 0:
                    perceptron.train(alpha, label, train_input[i])

        my_perceptrons.append(perceptron)

    return my_perceptrons


def perceptron_mini_batch(alpha, iterations, batch_size, train_input, train_label):
    my_perceptrons = []
    for digit in range(10):
        print(f"{digit} - mini batch")
        perceptron = MyPerceptron(digit)
        for _ in range(iterations):
            delta_w = np.zeros(784)
            delta_b = np.float64(0)
            current_batch = 0
            for i in np.arange(train_label.size):
                if train_label[i] == digit:
                    label = np.float64(1)
                else:
                    label = np.float64(-1)

                f = label * perceptron.test(train_input[i])

                if f <= 0:
                    delta_w += train_input[i] * label * alpha
                    delta_b += label * alpha

                current_batch += 1
                if current_batch == batch_size:
                    perceptron.add_batch(delta_w, delta_b)
                    delta_w = np.zeros(784)
                    delta_b = np.float64(0)
                    current_batch = 0

            if current_batch != 0:
                perceptron.add_batch(delta_w, delta_b)

        my_perceptrons.append(perceptron)

    return my_perceptrons


def validate_perceptron(name, my_perceptron, valid_input, valid_label):
    good = 0
    last = valid_input[0]
    label = 0
    ans_label = 0
    for i in np.arange(valid_label.size):
        value = []
        for digit in range(10):
            value.append(my_perceptron[digit].test(valid_input[i]))

        max_digit = value.index(max(value))
        if max_digit == valid_label[i]:
            good += 1
        else:
            last = valid_input[i]
            label = valid_label[i]
            ans_label = max_digit

    print(name)
    print(good, valid_label.size, good/valid_label.size * 100)
    plt.imshow(last.reshape(28, 28))
    plt.show()
    print(f"Last misclassified {label}, my label {ans_label}")


if __name__ == "__main__":
    train_input, train_label, validation_input, validation_label = get_data()

    perceptron_online = perceptron_online_training(0.05, 1, train_input, train_label)
    perceptron_batch = perceptron_mini_batch(0.05, 1, 500, train_input, train_label)

    validate_perceptron("Validation online training - training dataset: ", perceptron_online, train_input, train_label)
    validate_perceptron("validate mini-batch training - training dataset: ", perceptron_batch, train_input, train_label)

    validate_perceptron("Validate online training - validation dataset: ", perceptron_online,
                        validation_input, validation_label)
    validate_perceptron("Validate mini batch training - validation dataset: ", perceptron_batch,
                        validation_input, validation_label)
