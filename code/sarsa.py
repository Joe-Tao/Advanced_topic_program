import numpy as np
import random

class Agent:
    def __init__(self, state_space, action_space, epsilon, gamma, alpha):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.q = np.random.uniform(size = (state_space, action_space))
        # self.q = np.ones((state_space, action_space))
        self.name = "Sarsa"
        print("Running Sarsa")

    def observe(self, state1, state2, action1, action2, reward, done):
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * Q(s',a') - Q(s,a)]
        if done:
            self.q[state1, action1] += self.alpha * (
                reward - self.q[state1, action1]
            )
        else:
            self.q[state1, action1] += self.alpha * (
                reward + self.gamma * self.q[state2, action2] - self.q[state1, action1]
            )


    def act(self, state):

        if random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(self.action_space)
            # 1 - epsilon probability to greedy choose action
        else:
            # Break ties randomly
            if np.all(self.q[state, :]) == self.q[state, 0]:
                action = np.random.randint(self.action_space)
            else:
                action = np.argmax(self.q[state, :])
        # return action1 action2
        return action
    def getQ(self):
        return self.q