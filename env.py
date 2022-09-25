from xmlrpc.client import boolean
import numpy as np
from typing import List, Dict, Tuple
from enum import Enum



class State:
    def __init__(self, x: int, y: int, key: boolean):
        self.x = x
        self.y = y
        self.key = key

    @classmethod
    def from_int(index: int) -> State:
        pos = index // (env.H * env.W)
        return State(pos % env.W, pos // env.H, index % (env.H * env.W))

    def __index__(self):
        return self.key * (env.H * env.W) + self.y * env.W + self.x


class Cell:
    def __init__(self, x, y, is_goal, has_key, is_start):
        self.x = x
        self.y = y
        self.is_goal = is_goal
        self.has_key = has_key
        self.is_start = is_start

class Environment:
    def __init__(self, H: int, W: int, start_cell: Cell):
        self.grid : Cell = start_cell
        self.H = H
        self.W = W
        current_state: State = State(0,0,False)

    def __str__(self):
        pass

    def step(self, action) -> Tuple[State, float]:
        reward = -1

        if action is Action.UP:
            self.current_state.x -= 1
        elif action is Action.LEFT:
            self.current_state.y -= 1
        elif action is Action.DOWN:
            self.current_state.x += 1
        elif action is Action.RIGHT:
            self.current_state.x += 1

        elif action is Action.PICK_UP:
            cell = self.grid.get_cell(self.current_state.x, self.current_state.y)
            if cell.has_key:
                self.current_state.key = True
                cell.has_key = False

        elif action is Action.USE:
            cell = self.grid.get_cell(self.current_state.x, self.current_state.y)
            if cell.is_goal and self.key:
                reward = 0
        return self.current_state, reward


class Action(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3
    PICK_UP = 4
    USE = 5


class Agent:
    def __init__(self, nb_states: int, nb_actions: int, eps: float = 0.1):
        self.nb_action = nb_actions
        self.nb_states = nb_states
        self.eps = eps
        self.Q: np.array = np.random.rand(nb_states, nb_actions)

    def act(self, current_state: State):
        if np.random.rand() < self.eps:
            return np.random.randint(self.nb_actions)
        return np.argmax(self.Q[int(current_state)])

def loop():
    env = Environment()
    agent = Agent()
    rewards: List[float] = []
    while env.is_done():
        action = agent.act(env.get_state(agent))
        new_state, reward = env.step(action)
        rewards.append(reward)

