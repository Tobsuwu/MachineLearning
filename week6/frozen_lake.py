import gym
import numpy as np
import matplotlib.pyplot as plt


def manual_solution():
    """
    Manual solution asked in the handout. There are several minimum move (6) solutions and this is one of them.
    :return:
    """

    env = gym.make("FrozenLake-v1", is_slippery=False)
    env.reset()
    env.render()

    state, reward, done, info = env.step(1)
    env.render()
    state, reward, done, info = env.step(1)
    env.render()
    state, reward, done, info = env.step(2)
    env.render()
    state, reward, done, info = env.step(2)
    env.render()
    state, reward, done, info = env.step(1)
    env.render()
    state, reward, done, info = env.step(2)
    env.render()


def eval_policy(qtable_, num_of_episodes_, max_steps_, slippery):
    """
    Evaluation of the q-table. Copied from the lecture notes
    :param qtable_: the q_table to be evaluated
    :param num_of_episodes_: number of episodes
    :param max_steps_: maximum steps per episode
    :return:
    """

    rewards = []
    env = gym.make("FrozenLake-v1", is_slippery=slippery)

    for episode in range(num_of_episodes_):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps_):
            action = np.argmax(qtable_[state, :])
            new_state, reward, done, info = env.step(action)
            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    env.close()
    avg_reward = sum(rewards)/num_of_episodes_
    return avg_reward


def q_learning(slippery, nondeterministic):
    """
    Q_learning algorithm for problems a) and b) of the handout.
    :param slippery: Determines if the game is slippery aka deterministic or non-deterministic
    :param nondeterministic: Determines if the game uses a non-deterministic update rule in q_learning function
    :return: averages:
    """

    reward_best = -1000
    total_episodes = 1000
    max_steps = 100
    gamma = 0.9
    alpha = 0.5
    averages = []

    env = gym.make("FrozenLake-v1", is_slippery=slippery)
    env.reset()
    state_size = env.observation_space.n
    action_size = env.action_space.n

    # qtable = np.random.rand(state_size, action_size)  # initialize Random Q-table.
    qtable = np.zeros((state_size, action_size))
    for episode in range(total_episodes):
        state = env.reset()
        done = False
        reward_tot = 0
        for step in range(max_steps):
            action = np.random.randint(0, 3)  # Random action.
            new_state, reward, done, info = env.step(action)
            reward_tot += reward
            if nondeterministic:
                qtable[state, action] = qtable[state, action] + alpha*(reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
            else:
                qtable[state, action] = reward + gamma * np.max(qtable[new_state])
            state = new_state
            if done:
                break
        if reward_tot > reward_best:
            reward_best = reward_tot
            qtable_best = qtable
        if episode % 50 == 0:
            average = eval_policy(qtable_best, total_episodes, max_steps, slippery)
            averages.append(average)

    return averages


def plot(q_table_averages, episode_num):
    """
    Plot the different runs
    :param q_table_averages: array of 10 different q_table averages accros the runs
    :param episode_num: Number of episodes used in runs
    :return:
    """

    for i in range(len(q_table_averages)):
        plt.plot(episode_num, q_table_averages[i], label=f'Q-table run: {i+1}')

    plt.xlabel('Number of episodes run')
    plt.ylabel('Evaluation average (reward)')
    plt.title('Performance (average reward) of different Q-table runs')
    plt.legend(loc='upper left')
    plt.show()


def abc_problem(slippery, nondeterministic):
    """
    Problem a), b) and c) of the handout
    :param slippery: Determines if the game is slippery aka deterministic or non-deterministic
    :param nondeterministic: Determines if the game uses a non-deterministic update rule in q_learning function
    :return:
    """

    episode_num = np.linspace(0, 1000, 20)
    q_table_averages = []
    for i in range(10):
        average = q_learning(slippery, nondeterministic)
        q_table_averages.append(average)
    q_table_averages = np.asarray(q_table_averages)
    plot(q_table_averages, episode_num)


def main():

    # manual_solution()
    #abc_problem(False, False)
    #abc_problem(True, False)
    abc_problem(True, True)


main()
