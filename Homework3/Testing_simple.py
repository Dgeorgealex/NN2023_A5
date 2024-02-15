import random
import flappy_bird_gym
import pygame.time

from replay_memory import ReplayMemory, Transition
from frame_stack import FrameStack
from dqn_simple import DnqSimple, FRAMES
import torch
import utils

policy_net = DnqSimple()
nothing = torch.load("DQNSIMPLE/1000000")
policy_net.load_state_dict(nothing)

env = flappy_bird_gym.make("FlappyBird-v0")

def select_action(state_tensor):
    ans = policy_net(state_tensor)
    print(ans)
    return ans.max(1).indices.view(1, 1)


if __name__ == "__main__":

    clock = pygame.time.Clock()

    state = env.reset()
    state_tensor = utils.tensor_from_state(state)

    frames = FrameStack(FRAMES, "simple")
    frames.add_state(state_tensor)

    score = 0
    while True:
        env.render()
        action = select_action(frames.get_current_state())

        state, reward, done, info = env.step(action.item())

        score += reward

        print(state)
        state_tensor = utils.tensor_from_state(state)

        frames.add_state(state_tensor)

        clock.tick(15)

        if done:
            break

    print(score)

