import torch


class FrameStack:
    def __init__(self, num_frames, type):
        self.num_frames = num_frames
        if type == "rgb":
            self.current_state = torch.zeros(self.num_frames, 84, 84, device="cuda")
        else:
            self.current_state = torch.zeros(self.num_frames, 2, device="cuda")

    def add_state(self, state):
        self.current_state[:-1] = self.current_state[1:].clone()
        self.current_state[-1] = state.clone()

    def get_current_state(self):
        return self.current_state.clone().unsqueeze(0)

