import math
import random
from itertools import count
import flappy_bird_gym
import torch
from collections import namedtuple, deque
import pygame


env = flappy_bird_gym.make("FlappyBird-v0")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

clock = pygame.time.Clock()


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


class DQN(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.layer1 = torch.nn.Linear(n_observations, 128)
        torch.nn.init.zeros_(self.layer1.weight)
        torch.nn.init.zeros_(self.layer1.bias)

        self.layer2 = torch.nn.Linear(128, 128)
        torch.nn.init.zeros_(self.layer2.weight)
        torch.nn.init.zeros_(self.layer2.bias)

        self.layer3 = torch.nn.Linear(128, n_actions)
        torch.nn.init.zeros_(self.layer3.weight)
        torch.nn.init.zeros_(self.layer3.bias)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 100
GAMMA = 0.99
EPS_START = 0.01
EPS_END = 0.01
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            ans = policy_net(state)
            print(ans)
            return ans.max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

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
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if __name__ == "__main__":
    num_episodes = 1000

    for i_episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        for t in count():
            env.render()
            action = select_action(state)
            observation, reward, done, info = env.step(action.item())
            reward = torch.tensor([reward])

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

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

            clock.tick(60)

            if done:
                print(f"iteration: {i_episode}, reward {t}")
                break

    print("complete")