import random
import flappy_bird_gym
import pygame.time

from replay_memory import ReplayMemory, Transition
from frame_stack import FrameStack
from dqn_simple import DnqSimple, FRAMES
import torch
import utils

env = flappy_bird_gym.make("FlappyBird-v0")

policy_net = DnqSimple()
target_net = DnqSimple()
target_net.load_state_dict(policy_net.state_dict())

memory = ReplayMemory(20000)

LR = 1e-5
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)


MAX_ITERATIONS = 1000000

SAVE_INTERVAL = 5000

RANDOM_MOVES = 3000

UPDATE_COUNT = 1000

BATCH_SIZE = 32

GAMMA = 0.95

epsilon = 1
MIN_EPS = 0.01
EPS_DECAY = 0.99995

iterations = 1


def select_action(state_tensor):
    global epsilon, iterations

    sample = random.random()

    if iterations > RANDOM_MOVES:
        epsilon *= EPS_DECAY
        epsilon = max(epsilon, MIN_EPS)

    if sample > epsilon:
        with torch.no_grad():
            ans = policy_net(state_tensor)
            # print(ans)
            return ans.max(1).indices.view(1, 1)
    else:
        sample = random.random()
        if sample < 0.9:
            return torch.tensor([[0]])
        else:
            return torch.tensor([[1]])


def optimize_model():
    if len(memory) < RANDOM_MOVES:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    next_state_batch = torch.cat(batch.next_state)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1).values.detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10)
    optimizer.step()


# I will keep a history of 5 frames
def train():
    global iterations
    clock = pygame.time.Clock()
    episodes = 0
    maximum = 0
    while True:
        episodes += 1

        frames = FrameStack(FRAMES, "simple")
        state = env.reset()
        state_tensor = utils.tensor_from_state(state)

        frames.add_state(state_tensor)

        score = 0
        while True:
            # env.render()

            action = select_action(frames.get_current_state())

            state, reward, done, info = env.step(action.item())

            state_tensor = utils.tensor_from_state(state)

            score += reward
            reward = torch.tensor([reward])

            copy_of_current_state = frames.get_current_state()
            frames.add_state(state_tensor)

            # print(copy_of_current_state)
            # print(frames.get_current_state())
            # print()

            memory.push(copy_of_current_state, action, frames.get_current_state(), reward)

            optimize_model()

            if iterations % UPDATE_COUNT == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if iterations % SAVE_INTERVAL == 0:
                torch.save(target_net.state_dict(), f'DQNSIMPLE/{iterations}')

            if iterations == MAX_ITERATIONS:
                print("Training done")
                exit(0)

            iterations += 1

            if done:
                maximum = max(maximum, score)
                print(f"Episode {episodes} maximum score until now {maximum} now score {score} epsilon {epsilon} iterations {iterations}")
                break


if __name__ == "__main__":
    train()
