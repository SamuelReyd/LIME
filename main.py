from utils import *
from env import *
from agent import *
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


def plot_perf_live(returns, path_length, fig, hfig, ax0, ax1, ax2, ax3, Q, H, W):
    ax0.cla()
    ax1.cla()
    ax2.cla()
    ax3.cla()

    without_key_Q = Q.max(axis=1)[:Q.shape[0]//2].reshape(H, W)
    ax2.imshow(without_key_Q)
    for (j,i),label in np.ndenumerate(without_key_Q):
        ax2.text(i,j,f"{label:.3f}",ha='center',va='center')
        ax2.text(i,j,f"{label:.3f}",ha='center',va='center')
    # ax2.colorbar()
    ax2.set_title("Best agent value without key")

    with_key_Q = Q.max(axis=1)[Q.shape[0]//2:].reshape(H, W)
    ax3.imshow(with_key_Q)
    for (j,i),label in np.ndenumerate(with_key_Q):
        ax3.text(i,j,f"{label:.3f}",ha='center',va='center')
        ax3.text(i,j,f"{label:.3f}",ha='center',va='center')
    # ax3.colorbar()
    ax3.set_title("Best agent value with key")

    lines = []
    n = len(returns)
    plot_ids = np.linspace(0, n-1, min(n-1,100), dtype=int)
    lines += ax0.plot(plot_ids, np.array(returns)[plot_ids], label="returns", c="cyan", linestyle="--")
    ax0.set_ylabel("Value of the return")
    lines += ax1.plot(plot_ids, np.array(path_length)[plot_ids], label="nb steps", c="orange") 
    ax1.set_ylabel("Nb of steps")
    ax1.set_xlabel("Runs")

    ax1.legend(lines, [l.get_label() for l in lines], loc=0)
    fig.canvas.draw()
    hfig.update(fig)

def main(name, max_steps, num_run, learn=True, agent=None, agent_class=TabularAgent):
    fig, axes = plt.subplots(1,3)
    plt.rcParams["figure.figsize"] = (30,5)
    ax1, ax2, ax3 = axes
    ax0 = ax1.twinx()
    disp = display(fig, display_id=True)
    plt.tight_layout()
    plt.rcParams["figure.figsize"] = (30,5)

    env = Environment.crate_maze(name=name, max_steps=max_steps)
    if agent is None: agent = agent_class(env)
    agent.learning = learn

    all_states = []
    all_actions = []
    all_rewards = []
    returns = []
    path_length = []
    for i in tqdm(range(num_run)):
        states, actions, rewards, agent = env.rollout(agent, verbose=False, learn=learn)
        env.next_run()
        agent.next_run(env.current_state)
        all_states += [states]
        all_actions += [actions]
        all_rewards += [rewards]
        returns.append(sum(rewards))
        path_length.append(len(states))
        if i % (num_run // 50) == 0:
            plot_perf_live(returns, path_length, fig, disp, ax0, ax1, ax2, ax3, agent.Q, env.H, env.W)

    return agent



