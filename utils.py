from enum import Enum
import torch
from torch import nn
from typing import List

class State:
    dim = 3
    def __init__(self, x: int, y: int, key: bool, H: int, W: int):
        self.x = x
        self.y = y
        self.key = key
        self.H = H
        self.W = W


    def __str__(self):
        return f"State<{self.x}, {self.y}, {self.key}>"

    @staticmethod
    def from_int(index: int, H: int, W: int):
        pos = index // (H * W)
        return State(pos % W, pos // H, index % (H * W), H, W)

    def __index__(self):
        return self.key * (self.H * self.W) + self.y * self.W + self.x

class ConvNetwork(nn.Module):
    def __init__(self, env, embed_dim) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(env.cell_dim, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2)
        )
        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.2)
        )

        self.layer3 = torch.nn.Sequential(
            nn.Linear(32, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2)
        )

    def forward(self, input):
        h1 = self.layer1(input)
        h2 = self.layer2(h1)
        h2_bis = h2.view(h2.size(0), -1)
        y = self.layer3(h2_bis)
        return y

class MLPNetwork(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Dropout(0.2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, state):
        out = self.layer(state)
        return out

def get_state_vect(state:State) -> torch.FloatTensor:
    return torch.FloatTensor([state.x, state.y, state.key])

def get_state_vect_batch(states: List[State]) -> torch.FloatTensor:
    return torch.stack([get_state_vect(state) for state in states])

class Action(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3
    PICK_UP = 4
    # USE = 5