import numpy as np
from utils import *
import torch
import copy



class Agent:
    def __init__(self, env, eps: float = 0.5, step_size: float = 0.2, decay: float = 0.99):
        self.nb_actions: int = len(Action)
        self.nb_states: int = env.get_num_states()
        self.previous_state: State = env.current_state
        self.previous_action: Action = None
        self.step_size = step_size
        self.decay = decay
        self.eps: float = eps
        self.learning = True

    def next_run(self, start_state:State):
        self.previous_state = start_state
        self.previous_action: Action = None

    def feedback(self, reward: float, current_state: State, is_done: bool):
        raise NotImplementedError

    def act(self, current_state: State):
        Q_values = self.get_Q_values(current_state)
        if self.learning:
            if np.random.rand() < self.eps:
                action = np.random.randint(self.nb_actions)
            else:
                action = np.argmax(Q_values)
        else:
            action = np.argmax(Q_values)
        action = Action(action)
        self.previous_action = action
        return action

class TabularAgent(Agent):

    def __init__(self, env, eps: float = 0.5, step_size: float = 0.2, decay: float = 0.99):
        super().__init__(env, eps, step_size, decay)
        self.Q = np.zeros((self.nb_states, self.nb_actions))

    def feedback(self, reward: float, current_state: State, is_done: bool):
        self.Q[int(self.previous_state), self.previous_action.value] += self.step_size * (reward + 
                                                                           self.decay * 
                                                                                max([self.Q[int(current_state), a] for a in range(self.nb_actions)]) * 
                                                                                (not is_done) -
                                                                           self.Q[int(self.previous_state), self.previous_action.value])
        self.previous_state = current_state

    def get_Q_values(self, state: State):
        return self.Q[int(state)]

class DQNAgent(Agent):
    def __init__(self, env, eps: float = 0.5, step_size: float = 0.2, decay: float = 0.99):
        super().__init__(env, eps, step_size, decay)
        self.policy_net = MLPNetwork(State.dim, len(Action))
        self.opt = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.target_net = MLPNetwork(State.dim, len(Action))
        self.grad_every = 200
        self.buffer_max_size = 10000
        self.update_every = 10
        self.batch_size = 128
        self.step = 0


        self.previous_state = env.current_state
        self.buffer_replay = []

    def get_Q_values(self, state: State):
        return self.policy_net(get_state_vect(state).unsqueeze(0)).detach().numpy()

    def feedback(self, reward: float, current_state: State, is_done: bool):
        # Update the buffer and current state
        if len(self.buffer_replay) > self.buffer_max_size:
            del self.buffer_replay[0]
        self.buffer_replay.append((self.previous_state, self.previous_action, reward, current_state))

        self.previous_state = current_state
        
        # # Train the network
        # # Check if enough experience
        # if len(self.buffer_replay) < self.buffer_max_size: return

        # # Sample a batch
        # batch = np.random.choice(self.buffer_replay, size=self.batch_size, replace=True)


        # # Forward the values

        # # Compute and backward the loss
        # self.Q_network.train()
        # self.Q_network.zero_grad()

        # # Updates the network


        
        # if self.step % self.update_every == 0:
        #     self.target_net.load_state_dict(self.policy_net.state_dict())


    
