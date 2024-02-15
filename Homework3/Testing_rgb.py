import time

import flappy_bird_gym
import numpy as np
import pygame
from PIL import Image
import torch
import utils
from frame_stack_cpu import FrameStackCpu
from dqn_cnn import DqnCnn


device = "cpu"

policy_net = DqnCnn().to(device)
nothing = torch.load("DQNCNN/1700000")
policy_net.load_state_dict(nothing)


def select_action(state_tensor):
    ans = policy_net(state_tensor).to(device)
    print(ans)
    return ans.max(1).indices.view(1, 1)


def play_with_render(env):
    clock = pygame.time.Clock()
    score = 0

    frames = FrameStackCpu(5, "rgb")
    state = env.reset()
    state = utils.preprocess_state_image(state)
    state_tensor = utils.tensor_from_state(state).to(device)

    frames.add_state(state_tensor)

    pipe = 0
    while True:
        env.render()

        # Getting action:
        action = select_action(frames.get_current_state())
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #     if (event.type == pygame.KEYDOWN and
        #             (event.key == pygame.K_SPACE or event.key == pygame.K_UP)):
        #         action = 1
        # Processing:
        state, reward, done, info = env.step(action.item())
        state = utils.preprocess_state_image(state)
        state_tensor = utils.tensor_from_state(state).to(device)

        frames.add_state(state_tensor)

        score += reward

        if reward == 1:
            pipe += 1
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        time.sleep(1/30)

        if done:
            env.render()
            time.sleep(0.5)
            break

    print(f"{score} {pipe}")


if __name__ == "__main__":
    flappy_env = flappy_bird_gym.make("FlappyBird-rgb-v0")

    play_with_render(env=flappy_env)

    flappy_env.close()
