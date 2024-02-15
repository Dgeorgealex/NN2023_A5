import torch.nn as nn

FRAMES = 5


class DnqSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_input = nn.Linear(FRAMES * 2, 128)
        self.fc_hidden1 = nn.Linear(128, 128)
        self.fc_hidden2 = nn.Linear(128, 128)  # New hidden layer with the same dimensions
        self.fc_output = nn.Linear(128, 2)
        self.relu = nn.ReLU()

        # self.initialize_weights()

    def initialize_weights(self):
        for layer in [self.fc_input, self.fc_hidden1, self.fc_hidden2, self.fc_output]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = x.view(-1, FRAMES * 2)
        x = self.relu(self.fc_input(x))
        x = self.relu(self.fc_hidden1(x))
        x = self.relu(self.fc_hidden2(x))  # Apply the new hidden layer
        x = self.fc_output(x)
        return x
