
import numpy as np
import random

class Agent:
    def __init__(self, state_space, action_space, epsilon, gamma, alpha):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.q_a = np.random.uniform(size = (state_space, action_space))
        self.q_b = np.random.uniform(size = (state_space, action_space))
        # self.q_a = np.ones((state_space, action_space))
        # self.q_b = np.ones((state_space, action_space))

        self.name = "Double Q learning"
        print("Running Double Q learning")

    def observe(self, action1, state1, state2, reward, done):

        if done:
            # run a
            if random.uniform(0, 1) < 0.5:
                self.q_a[state1, action1] += self.alpha * (
                        reward - self.q_a[state1, action1]
                )
            else: # run b
                self.q_b[state1, action1] += self.alpha * (
                        reward - self.q_b[state1, action1]
                )

        else:
            if random.uniform(0, 1) < 0.5:
                self.q_a[state1, action1] += self.alpha * (
                        reward + self.gamma * self.q_b[state2, np.argmax(self.q_a[state2])] - self.q_a[state1, action1]
                )
            else:
                self.q_b[state1, action1] += self.alpha * (
                        reward + self.gamma * self.q_a[state2, np.argmax(self.q_b[state2])] - self.q_b[state1, action1]
                )





    def act(self, state):
        q_table = self.q_a[state, :] + self.q_b[state, :]
        if random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(self.action_space)
            # 1 - epsilon probability to greedy choose action
        else:
            # Break ties randomly
            if np.all(q_table) == q_table[0]:
                action = np.random.randint(self.action_space)
            else:
                action = np.argmax(q_table)

        return action

    def getQ(self):
        return self.q_a + self.q_b