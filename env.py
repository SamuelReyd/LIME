from xmlrpc.client import boolean
import numpy as np
import json
from typing import List, Dict, Tuple
from enum import Enum
from agent import *






class Cell:
    def __init__(self, x, y, walls, is_goal=False, has_key=False, is_start=False):
        self.x = x
        self.y = y
        self.is_goal = is_goal
        self.has_key = has_key
        self.is_start = is_start
        self.walls = walls
    
    def has_wall(self, direction):
        return self.walls[direction.value]

class Environment:

    def __init__(self, H: int, W: int, cells: List[List[State]], max_steps:int):
        self.grid : List[List[Cell]] = cells
        self.H = H
        self.W = W
        self.start_cell = [cell for line in cells for cell in line if cell.is_start][0]
        self.current_state: State = State(self.start_cell.x, self.start_cell.y, False, H, W)
        self.goal_states: List[State] = [State(cell.x, cell.y, True, H, W) for line in cells for cell in line if cell.is_goal]
        self.n_step = 0
        self.is_done = False
        self.max_steps = max_steps
        self.cell_dim = 6 # 4 walls, key and goal

        # TODO: Testing if cells are compatibles:

    def __str__(self):
        S = ""
        # Plafond
        for line in self.grid:
            up_line = []
            bottom_line = []
            current_line = []
            for cell in line:
                up_line.append("---" if cell.walls[0] else "   ")
                current_line.append("|" if cell.walls[1] else " ")
                if cell.x == self.current_state.x and cell.y == self.current_state.y:
                    current_line.append("P")
                elif cell.is_start:
                    current_line.append("S")
                elif cell.is_goal:
                    current_line.append("G")
                elif cell.has_key:
                    current_line.append("K")
                else:
                    current_line.append("E")
                current_line.append("|" if cell.walls[3] else " ")
                bottom_line.append("---" if cell.walls[2] else "   ")
            S = S + "\n".join(["".join(l) for l in (up_line, current_line, bottom_line)]) + "\n"
        S = S + "\b"
        return S

    def next_run(self):
        self.current_state: State = State(self.start_cell.x, self.start_cell.y, False, self.H, self.W)
        self.n_step = 0
        self.is_done = False

    def rollout(self, agent, verbose:int=0, learn:bool=True) -> Tuple[List[State], List[Action], List[float], Agent]:
        rewards: List[float] = []
        states: List[State] = []
        actions: List[Action] = []
        while not self.is_done:
            states.append(self.current_state)
            action = agent.act(self.current_state) # get the agent action
            reward = self.step(action, verbose) # the environment evolve based on the agent action and returns a reward
            if learn: agent.feedback(reward, self.current_state, self.is_done)
            rewards.append(reward)
            actions.append(action)
            if verbose == 1: 
                print(self.n_step, states[-1], action, reward, '->', self.current_state)
            if verbose == 2: 
                print(self)
                print(action)
                print("_"*30)
                print()
        return states, actions, rewards, agent

    def get_cell(self, x:int, y:int) -> Cell:
        return self.grid[y][x]

    def get_current_cell(self) -> Cell:
        return self.get_cell(self.current_state.x, self.current_state.y)

    def step(self, action, verbose=False) -> Tuple[State, float]:
        if verbose == 1: print("  Acting...")
        if verbose == 1: print("  Cell:", self.get_current_cell().walls)
        if verbose == 1: print("  Move:", action)
        if action.value in range(4):
            if verbose == 1: print("  Autorized:", not self.get_current_cell().has_wall(action))
        if self.n_step > self.max_steps: 
            self.is_done = True
            return -200
        self.n_step += 1
        reward = -1
        x = self.current_state.x
        y = self.current_state.y
        key = self.current_state.key

        cell = self.get_current_cell()
        if cell.is_goal and self.current_state.key:
            reward = 0
            self.is_done = True

        elif action is Action.UP and not self.get_current_cell().has_wall(Action.UP):
            if verbose == 1: print("  Moved to:", x, y)
            y -= 1
        elif action is Action.LEFT and not self.get_current_cell().has_wall(Action.LEFT):
            if verbose == 1: print("  Moved to:", x, y)
            x -= 1
        elif action is Action.DOWN and not self.get_current_cell().has_wall(Action.DOWN):
            if verbose == 1: print("  Moved to:", x, y)
            y += 1
        elif action is Action.RIGHT and not self.get_current_cell().has_wall(Action.RIGHT):
            if verbose == 1: print("  Moved to:", x, y)
            x += 1

        elif action is Action.PICK_UP:
            cell = self.get_current_cell()
            if cell.has_key:
                key = True

        # elif action is Action.USE:
        #     cell = self.get_current_cell()
        #     if cell.is_goal and self.current_state.key:
        #         reward = 0
        #         self.is_done = True
        #         if verbose == 1: print("Done !")
        self.current_state = State(x, y, key, self.H, self.W)
        return reward

    def get_num_states(self):
        return self.H * self.W * 2

    def to_vect(self):
        x = []
        for line in self.grid:
            x_line = []
            for cell in line:
                x_line.append(cell.walls + [int(cell.has_key), int(cell.is_goal)])
            x.append(x_line)
        return np.rollaxis(np.array(x, dtype = float), -1)

    @staticmethod
    def crate_maze(name = "simple_maze", max_steps=10000):
        with open(f"{name}.json") as file:
            maze = json.load(file)
        grid = []
        for i in range(len(maze["walls"])):
            line = []
            for j in range(len(maze["walls"][0])):
                line.append(Cell(j,i, maze["walls"][i][j]))
            grid.append(line)
        grid[maze["start"][0]][maze["start"][1]].is_start = True
        grid[maze["key"][0]][maze["key"][1]].has_key = True
        grid[maze["end"][0]][maze["end"][1]].is_goal = True
        return Environment(len(grid), len(grid[0]), grid, max_steps)

    def show_run(self, states, file_out=None):
        S = ""
        for state in states:
            self.current_state = state
            if file_out is None: print(self)
            else:
                S += self.__str__()
        if file_out:
            with open(file_out, "w") as file:
                file.write(S)