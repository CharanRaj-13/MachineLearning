'''
pip install numpy
pip install matplotlib
'''

import numpy as np
import matplotlib.pyplot as plt

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)  # Count of rewards 1
        self.failures = np.zeros(n_arms)  # Count of rewards 0

    def select_arm(self):
        sampled_theta = [
            np.random.beta(self.successes[i] + 1, self.failures[i] + 1)
            for i in range(self.n_arms)
        ]
        return np.argmax(sampled_theta)

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1


# Simulated environment for the multi-armed bandit
class BanditEnvironment:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def pull_arm(self, arm):
        return 1 if np.random.rand() < self.probabilities[arm] else 0


# Parameters
n_arms = 3
probabilities = [0.2, 0.5, 0.75]  # Probabilities of success for each arm
n_trials = 1000

# Initialize the environment and the agent
env = BanditEnvironment(probabilities)
agent = ThompsonSampling(n_arms)

# Run Thompson Sampling
rewards = []
chosen_arms = []

for trial in range(n_trials):
    chosen_arm = agent.select_arm()
    reward = env.pull_arm(chosen_arm)
    agent.update(chosen_arm, reward)

    rewards.append(reward)
    chosen_arms.append(chosen_arm)

# Visualization
cumulative_rewards = np.cumsum(rewards)
optimal_arm = np.argmax(probabilities)
optimal_arm_selections = [1 if arm == optimal_arm else 0 for arm in chosen_arms]
cumulative_optimal_selections = np.cumsum(optimal_arm_selections)

plt.figure(figsize=(12, 6))

# Cumulative Rewards Plot
plt.subplot(1, 2, 1)
plt.plot(cumulative_rewards)
plt.title("Cumulative Rewards")
plt.xlabel("Trials")
plt.ylabel("Cumulative Reward")

# Optimal Arm Selection Plot
plt.subplot(1, 2, 2)
plt.plot(cumulative_optimal_selections)
plt.title("Optimal Arm Selections")
plt.xlabel("Trials")
plt.ylabel("Cumulative Optimal Selections")

plt.tight_layout()
plt.show()

print(f"Total Reward: {np.sum(rewards)}")
print(f"Optimal Arm Selections: {cumulative_optimal_selections[-1]}/{n_trials}")
