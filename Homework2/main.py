import torch
import pickle
import gzip
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt


def read_data():
    with gzip.open("mnist.pkl.gz", "rb") as fd:
        train_set, valid_set, test_set = pickle.load(fd, encoding="latin")

    train_data = MyDataset(train_set)
    valid_data = MyDataset(valid_set)

    return train_data, valid_data


class MyDataset(Dataset):
    def __init__(self, data):
        self.samples = torch.tensor(data[0])
        self.labels = torch.zeros(len(data[1]), 10)
        for i in range(len(data[1])):
            self.labels[i, data[1][i]] = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class MyNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.first_layer = torch.nn.Linear(784, 500, bias=True)
        self.second_layer = torch.nn.Linear(500, 10, bias=True)

        torch.nn.init.kaiming_normal_(self.first_layer.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.second_layer.weight)

        self.layers = torch.nn.Sequential(self.first_layer, torch.nn.ReLU(),
                                          self.second_layer)

    def forward(self, inputs):
        return self.layers(inputs)


def train_model(train_data):
    start_time = time.time()

    # best as time/result =  0.5 - batch = 100 - 30 epoch
    model = MyNN()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    softmax = torch.nn.Softmax(dim=1)       # needs to be like this because of mini-batch
    dataloader = DataLoader(train_data, batch_size=100, shuffle=True)

    for i in range(30):     # number of epochs
        for batch_id, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            outputs_softmax = softmax(outputs)
            loss = loss_function(outputs_softmax, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(i)

    end_time = time.time()
    print(f"Training done: {end_time - start_time}")
    return model


def test_model(model, valid_data):
    ans = [{"tp": 0, "fp": 0, "tn": 0, "fn": 0} for _ in range(10)]
    good = 0
    bad = []
    for test_id, (inputs, labels) in enumerate(valid_data):
        outputs = model(inputs)

        my_digit = torch.argmax(outputs).item()
        true_digit = torch.argmax(labels).item()

        if my_digit == true_digit:
            good += 1
            ans[true_digit]["tp"] += 1

        if my_digit != true_digit:
            ans[my_digit]["fp"] += 1
            ans[true_digit]["fn"] += 1
            bad.append((inputs, true_digit, my_digit))

        for i in range(10):
            if i != true_digit and i != my_digit:
                ans[i]["tn"] += 1

    print(f"General accuracy: {good / len(valid_data) * 100}")
    for digit in range(10):
        precision = ans[digit]["tp"] / (ans[digit]["tp"] + ans[digit]["fp"])
        recall = ans[digit]["tp"] / (ans[digit]["tp"] + ans[digit]["fn"])
        f1 = 2 * precision * recall / (precision + recall)
        print(f"{digit}: Precision: {precision:.5f}; Recall: {recall:.5f}; f1-score: {f1:.5f}")

    for i in range(5):
        plt.imshow(bad[i][0].reshape(28, 28))
        plt.show()
        print(f"True label: {bad[i][1]} - My label: {bad[i][2]}")


def main():
    train_data, valid_data = read_data()

    # model = torch.load('MyModel.pt')

    model = train_model(train_data)
    torch.save(model, 'MyModel.pt')

    test_model(model, train_data)
    test_model(model, valid_data)


if __name__ == "__main__":
    main()