import gym
import torch
from torch import nn
from collections import deque
import cv2
import numpy as np
import os

cv2.ocl.setUseOpenCL(False)

device = "cpu"


class model(nn.Module):
    def __init__(self, n_frame, n_action, noisy, device):
        super(model, self).__init__()
        self.noisy = noisy
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_frame, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Dynamically calculate the flattened size
        dummy_input = torch.zeros(1, n_frame, 84, 84)  # Example input size
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
        flattened_size = dummy_output.numel()

        self.fc = nn.Linear(flattened_size, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)

        self.device = device

        # Initialize weights for convolutional layers
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        if self.noisy:
            self.reset_noise()
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - adv.mean())

        return q


q = model(4, 12, False, device)
q.load_state_dict(torch.load("3954.pth", map_location=torch.device('cpu')))
q.to(device)


def normalize(x):
    return x.to(torch.float32) / 255.0


# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.counter = 0
        self.action = None
        self.frames = deque(maxlen=4)
        self._obs_buffer = np.zeros((2,) + (240, 256, 3), dtype=np.uint8)
        self.cnt = 0

        self.mask = np.ones((84, 84), dtype=np.int8)
        regions_to_mask = [(7, 5, 17, 6), (28, 7, 11, 4), (66, 5, 10, 6)]
        for region in regions_to_mask:
            x, y, w, h = region
            self.mask[y : y + h, x : x + w] = 0

    def to_frame(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = cv2.bitwise_and(frame, frame, mask=self.mask)
        return frame

    def update_action(self):
        state = np.stack(self.frames, axis=0)
        state = torch.from_numpy(state).unsqueeze(0)
        s_expanded = normalize(state).to(device)
        with torch.no_grad():
            self.action = q(s_expanded).argmax().item()

    def act(self, observation):
        # <class 'numpy.ndarray'> (240, 256, 3) uint8
        # print(type(observation), observation.shape, observation.dtype)

        if len(self.frames) < 4:  # FrameStack reset
            frame = self.to_frame(observation)
            while len(self.frames) < 4:
                self.frames.append(frame)
            self.update_action()
            return self.action

        else:
            # max and skip frames
            if self.counter == 2 or self.counter == 3:
                self._obs_buffer[self.counter - 2] = observation

            if self.counter == 3:
                self.frames.append(self.to_frame(self._obs_buffer.max(axis=0)))
                # filename = f"rec/{self.cnt}.png"
                # if os.path.exists(filename):
                #    exit()
                # cv2.imwrite(filename, self.frames[3])
                # self.cnt += 1

                self.update_action()
                self.counter = (self.counter + 1) % 4
                return self.action
            else:
                self.counter = (self.counter + 1) % 4
                return self.action
