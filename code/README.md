# Reinforcement Learning assignment

The goal of this assignment is to create your own reinforcement learning agents.
This GitHub repo contains both the instructions and the files needed for the assignment.
The assignment will use the [gymnasium](https://gymnasium.farama.org/) structure, 
I suggest you familiarize yourselves with this before getting started.

**agent.py** is a template agent for you to fill in.

**riverswim.py** contains the custom environment RiverSwim. '''python3 run_experiment.py --env riverswim:RiverSwim''' will run this environment.

**run_experiment.py** serves as a template for how to run your experiments, it allows you to load different agents and 
environments you might create by running. You can either call the existing gymnasium environments or ones you have created yourself.

## General instructions
You will need the gym and numpy libraries. 

You should implement the algorithms yourselves, not using implementations by anybody else. 
****
## Tasks
The tasks consist of describing, implementing, and analysing Q-learning, Double Q-learning, SARSA and Expected SARSA.
### 1. Describe the algorithms
Briefly describe the Q-learning, Double Q-learning, SARSA and Expected SARSA algorithm and how each of them work.

### 2. Run and plot rewards
Implement Q-learning, Double Q-learning, SARSA and Expected SARSA agents with 5% epsilon greedy policy (95% greedy action) and $\gamma = 0.95$.
For every agent, run experiments on both RiverSwim (included locally in this repository) and [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).  
Plot the moving average of the rewards while your agent learns, either by episode or step depending on the environment, averaged over 5 runs (i.e., restarting the training 5 times). 
Include error-bars (or something similar) indicating the 95% confidence intervals calculated from your variance of the 5 runs.

### 3. Visualize Q-values and greedy policy
For every combination of agent and enviornment in Task 2., visualize the Q-values for each state (make sure it is easy to interpret, i.e., not just a table). Draw/visualize the greedy policy obtained from the Q-values. Does it reach the optimal policy? Briefly describe and motivate the optimal policy. 

### 4. Initialization of Q-values
Investigate how the initialization of the Q-values affects how well an agent learns to solve RiverSwim and FrozenLake-v1. For instance, initializing Q-tables optimistically (larger than you expect the true table to be, not with zeroes). How does it affect the rewards during training and the greedy policy obtained from the Q-values? How does it affect the exploration? Clearly explain and display evidence supporting your arguments. 

### 5. Discussion
What are your conclusions? How do the agents perform? Discuss if the behaviour is expected based on what you have learned in the course. 

## Grading
The grading will be based on:
1. **Your report**, how well you describe, analyse, and visualize your results and how clear your text is.
2. **Correctness**, how well your solutions perform as expected.

## Tips
- Note that some environments have a **done** flag to indicate that the episode is finished. Make sure you take this into account. 
   - For example, you do not want any future rewards from the terminal state and no transition from terminal state to starting state. 
- The explorative policy might look bad due to the epsilon-exploration, good to check if the Q-table seems to give a good policy. You can always compare with the Bellman equation.
- Since FrozenLake is slippery, it might look like good policy is sub-optimal. There are some subtle dynamics coming from that the agent can only slip perpendicularly to the direction of the action that can be worth thinking about.
- The number of steps in the run_experiments is just an example, you will probably need more steps.
- You will need to have $\alpha$ "sufficiently small" otherwise there will be high variance ($\alpha=1$ would mean that you ignore the old value and overwrite it).

## Submission
Make sure to include both your code and your report when you submit your assignment. Your report should be submitted as a PDF.

