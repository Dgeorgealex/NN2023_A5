import random
from itertools import count
import flappy_bird_gym
import torch
from collections import namedtuple, deque
import pygame
import cv2
from PIL import Image

clock = pygame.time.Clock()

BATCH_SIZE = 30
GAMMA = 0.99

EPS = 0.05

TAU = 1e-2
LR = 1e-3

steps_done = 0
counter = 0

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env = flappy_bird_gym.make("FlappyBird-rgb-v0")


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


memory = ReplayMemory(300)


class DqnCnn(torch.nn.Module):
    def __init__(self):
        super(DqnCnn, self).__init__()
        # intra 1 x 128 x 128
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)  # 16 x 64 x 64
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 16 x 32 x 32

        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # 32 * 16 * 16
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 32 * 8 * 8

        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 64 x 4 x 4
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 64 x 2 x 2

        self.fc1 = torch.nn.Linear(128, 128)
        torch.nn.init.zeros_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

        self.relu3 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(128, 2)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


policy_net = DqnCnn()
target_net = DqnCnn()
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)


def preprocess_image(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    normalized = resized
    state_tensor = torch.tensor(normalized)
    transformed_tensor = state_tensor.unsqueeze(0).repeat(4, 1, 1)
    return transformed_tensor


# def print_obs(state):
#     gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
#     resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
#     normalized = resized
#     img = Image.fromarray(normalized)
#     img.show()
#     img.close()


def select_action(state):
    global steps_done
    global counter
    sample = random.random()
    # eps_threshold = (EPS_END + (EPS_START - EPS_END) *
    #                  math.exp(-1. * steps_done / EPS_DECAY))        # I should change this
    steps_done += 1
    if sample > EPS:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        counter += 1
        return torch.tensor([[env.action_space.sample()]])


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values.detach()


    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10)
    optimizer.step()


def train():
    num_episodes = 1000

    for i_episode in range(num_episodes):
        state = env.reset()
        state = preprocess_image(state)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        a = 0
        for t in count():
            # env.render()
            if t <= 60:
                if t % 20 == 0:
                    observation, reward, done, info = env.step(1)
                else:
                    observation, reward, done, info = env.step(0)
                continue

            action = select_action(state)
            # print(f"{t} and {action.item()}")

            observation, reward, done, info = env.step(action.item())

            reward = (torch.tensor([reward]))
            observation = preprocess_image(observation)
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            a += reward

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_((1.0 - TAU) * target_param.data + TAU * policy_param.data)

            # clock.tick(80)

            if done:
                print(counter)
                print(f"iteration: {i_episode}, reward {a}")
                break

    print("complete")


if __name__ == "__main__":
    train()
