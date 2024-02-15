import random
import time

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_state_image(state):    # 512 x 288 = modify such that it eliminates the
    state = np.moveaxis(state, source=1, destination=0)
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = state[:340, :]
    # state = state[::3, ::3]     # down sampling - i don't want to do it
    state = cv2.resize(state, (84, 84))

    _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
    state = state / 255.0
    return state


def tensor_from_state(state):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    return state_tensor


def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device
