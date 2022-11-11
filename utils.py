from enum import Enum

class State:
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

class Action(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3
    PICK_UP = 4
    # USE = 5