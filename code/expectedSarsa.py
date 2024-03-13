import random
import numpy as np

class Agent:
    def __init__(self, state_space, action_space, epsilon, gamma, alpha):
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.q = np.random.uniform(size = (state_space,action_space))
        # self.q = np.ones((state_space, action_space))
        self.name = "Expected Sarsa"
        print("Running Expected Sarsa")

    def observe(self, action1, state1, state2, reward, done):
        # Update Q(S, A) = Q(S, A) + alpha(reward + (pi * Q(S_, A_) - Q(S, A))
        probabilities = np.ones(self.action_space) * self.epsilon / self.action_space
        probabilities[np.argmax(self.q[state2,:])] += 1 - self.epsilon
        expected_q = np.sum(probabilities * self.q[state2])
        difference = reward + self.gamma * expected_q
        if done:
            self.q[state1,action1] += self.alpha * (reward - self.q[state1,action1])
        else:
            self.q[state1,action1] += self.alpha * (difference - self.q[state1,action1])


    def act(self, state):
        explore_exploit_tradeoff = random.uniform(0, 1)
        if explore_exploit_tradeoff < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            values = self.q[state]
            max_value = np.max(values)
            actions = np.where(values == max_value)[0]
            action = np.random.choice(actions)
        return action

    def getQ(self):
        return self.q