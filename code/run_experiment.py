import argparse
from collections import deque

import gymnasium as gym
import importlib.util
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)


try:
    env = gym.make(
        args.env, is_slippery=True,
    )
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)


# Parameters
gamma = 0.95
epsilon = 0.05
alpha = 0.05
max_steps = 100
total_episodes = 10000
n_runs = 5
window_size = 2500
agent_name = 'agent'


rewards = np.zeros((n_runs, total_episodes))
action_dim = env.action_space.n
state_dim = env.observation_space.n

def moving_mean_rewards(arr, window_size):
    i = 0
    moving_averages = []

    while i < (len(arr) - window_size + 1):
        window_average = np.sum(arr[i:i + window_size]) / window_size
        moving_averages.append(window_average)

        i += 1

    return moving_averages


def run_experiment():
    for run in range(n_runs):
        agent = agentfile.Agent(state_dim, action_dim, epsilon, gamma, alpha)
        global agent_name
        agent_name = agent.name
        run_rewards = []
        for episode in range(total_episodes):
            done = False
            state1, info = env.reset()
            episode_reward = 0
            step = 0
            while step < max_steps:
                # env.render()
                step += 1
                action1 = agent.act(state1)  # your agent here (this currently takes random actions)
                state2, reward, done, truncated, info = env.step(action1)
                agent.observe(action1, state1, state2, reward, done)
                episode_reward += reward

                state1 = state2
                if done:
                    break
            run_rewards.append(episode_reward)
            # rewards[run, episode] = np.mean(run_rewards)
            rewards[run, episode] = episode_reward

    q_table = agent.getQ()
    if args.env == "riverswim:RiverSwim":
        draw_policy(q_table, 1, 6)
    else:
        draw_policy(q_table, 4, 4)
    visualize_q_table(q_table)

    env.close()


def run_sarsa_experiment():
    for run in range(n_runs):
        agent = agentfile.Agent(state_dim, action_dim, epsilon, gamma, alpha)
        global agent_name
        agent_name = agent.name
        run_rewards = []

        for episode in range(total_episodes):
            done = False
            state1, info = env.reset()
            episode_reward = 0
            action1 = agent.act(state1)
            step = 0
            while step < max_steps:
                # env.render()
                step += 1
                state2, reward, done, truncated, info = env.step(action1)
                action2 = agent.act(state2)
                episode_reward += reward
                agent.observe(state1, state2, action1, action2, reward, done)
                state1 = state2
                action1 = action2
                if done:
                    break
            run_rewards.append(episode_reward)
            # rewards[run, episode] = np.mean(run_rewards)
            rewards[run, episode] = episode_reward

    q_table = agent.getQ()
    if args.env == "riverswim:RiverSwim":
        draw_policy(q_table, 1, 6)
    else:
        draw_policy(q_table, 4, 4)
    visualize_q_table(q_table)

    env.close()

def draw_policy(q_table, map_x, map_y):
    directions = None
    if map_x == 4:
        directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    else:
        directions = {0: "←", 1: "→"}
    q_val_max = q_table.max(axis=1).reshape(map_x, map_y)
    q_best_action = np.argmax(q_table, axis=1).reshape(map_x, map_y)
    q_directions = np.empty(q_best_action.flatten().shape, dtype=str)

    for i, value in enumerate(q_best_action.flatten()):
        q_directions[i] = directions[value]

    q_directions = q_directions.reshape(map_x, map_y)
    sns.heatmap(
        q_val_max,
        annot=q_directions,
        fmt="",
        cmap=sns.color_palette("Greens", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title=f"Policy in Agent: {agent_name}\n In environment: {args.env}")
    plt.savefig(f'slippery/policy/{agent_name}_{args.env}.jpg')
    plt.show()


def visualize_q_table(q_table):
    state_num = q_table.shape[0]
    action_num = q_table.shape[1]
    q_grid = np.zeros((state_num, action_num))

    for state in range(state_num):
        for action in range(action_num):
            q_grid[state, action] = q_table[state][action]

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title(f"Q table in Agent: {agent_name}\n In environment: {args.env}")
    heatmap = ax.imshow(q_grid, cmap="Greens")

    # Set the axis labels
    ax.set_xticks(np.arange(action_num))
    ax.set_yticks(np.arange(state_num))
    # ax.set_xticklabels(np.arange(action_num))
    if state_num == 16:
        ax.set_xticklabels(["Left", "Down", "Right", "Up"])
    else:
        ax.set_xticklabels(["Left", "Right"])
    ax.set_yticklabels(np.arange(state_num))
    ax.set_xlabel("Action")
    ax.set_ylabel("States")

    # Add a colorbar
    cbar = ax.figure.colorbar(heatmap)
    plt.savefig(f'slippery/qtable/{agent_name}_{args.env}.jpg')
    # Display the plot
    plt.show()


def draw_moving_average():
    moving_rewards = np.zeros((n_runs,total_episodes-window_size+1 ))

    for run in range(len(rewards)):
        moving_average = moving_mean_rewards(rewards[run], window_size)
        moving_rewards[run] = moving_average


    mean = moving_rewards.mean(axis=0)
    ci = np.array(
        [st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a)) for a in moving_rewards.T]
    )
    plt.figure(figsize=(15,10))
    plt.fill_between(
        range(len(mean)),
        ci[:, 0],
        ci[:, 1],
        alpha=0.2,
        color="green",
        label=r"95% confidence interval",
    )
    for i, xs in enumerate(moving_rewards):
        plt.plot(xs, alpha=0.2, label=f"Run no.{1+i}")
    plt.plot(mean, label="Average of 5 runs", color = 'red')
    plt.title(f"Agent: {agent_name}  Environment: {args.env}", fontsize = 20)
    plt.xlabel("Episodes")
    plt.ylabel("Moving Average Rewards")
    plt.legend()
    plt.plot()
    plt.savefig(f'slippery/mean_moving/{agent_name}_{args.env}.jpg')
    plt.show()

if args.agentfile == 'sarsa.py':
    run_sarsa_experiment()
else:
    run_experiment()

draw_moving_average()
