import numpy as np
from utils import *
import torch



class Agent:
    def __init__(self, env, eps: float = 0.5, step_size: float = 0.2, decay: float = 0.99):
        self.nb_action: int = env.get_num_states()
        self.nb_states: int = len(Action)
        self.previous_state: State = env.current_state
        self.previous_action: Action = None
        self.Q = np.zeros((self.nb_states, self.nb_actions))
        self.step_size = step_size
        self.decay = decay
        self.eps: float = eps
        self.learning = True

    def next_run(self, start_state:State):
        self.previous_action = start_state
        self.previous_action: Action = None

    def feedback(self, reward: float, current_state: State, is_done: bool):
        raise NotImplemented

    def act(self, current_state: State):
        if self.learning:
            if np.random.rand() < self.eps:
                action = np.random.randint(self.nb_action)
            else:
                action = np.argmax(self.Q[int(current_state)])
        else:
            action = np.argmax(self.Q[int(current_state)])
        action = Action(action)
        self.previous_action = action
        return action

class TabularAgent(Agent):
    def feedback(self, reward: float, current_state: State, is_done: bool):
        self.Q[int(self.previous_state), self.previous_action.value] += self.step_size * (reward + 
                                                                           self.decay * 
                                                                                max([self.Q[int(current_state), a] for a in range(self.nb_action)]) * 
                                                                                (not is_done) -
                                                                           self.Q[int(self.previous_state), self.previous_action.value])
        self.previous_state = current_state

class DQNAgent(Agent):
    def __init__(self, env, eps: float = 0.5, step_size: float = 0.2, decay: float = 0.99):
        super().__init__(env, eps, step_size, decay)
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(env.cell_dim, 5),
            torch.nn.Conv2d(5, 5)
        )

    def feedback(self, reward: float, current_state: State, is_done: bool):
        self.Q[int(self.previous_state), self.previous_action.value] += self.step_size * (reward + 
                                                                           self.decay * 
                                                                                max([self.Q[int(current_state), a] for a in range(self.nb_action)]) * 
                                                                                (not is_done) -
                                                                           self.Q[int(self.previous_state), self.previous_action.value])
        self.previous_state = current_state
    
