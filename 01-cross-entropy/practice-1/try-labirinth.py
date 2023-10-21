import gym
import gym_maze
import time
import numpy as np


class RandomAgent():
    def __init__(self, action_n):
        self.action_n = action_n
        self.trajectory = {"action": [], "observation": []}
        self.list_trajectories = {"trajectory":[], "reward": []}
        self.env = gym.make('maze-sample-5x5-v0')

    def get_action(self):
        """Returns a random action from the available actions."""
        return np.random.randint(0, self.action_n)
    
    def put_step_in_trajectory(self, action, state):
        self.trajectory["action"].append(action)
        self.trajectory["observation"].append(state)
        return None
    
    def get_state(self, selfposition: tuple[int, int], max_dimension: int = 5) -> int:
        """Returns a unique state identifier from a given position in the maze.

        Args:
            position: A tuple of two integers representing the agent's current 
                    position in the maze.
            max_dimension: The maximum dimension of the maze.

        Returns:
            An integer representing the unique state identifier.
        """
        return selfposition[0] + max_dimension * selfposition[1]
    
    def sample_one_tajectory(self, num_steps=1000, visualize=False):
        observation, info = self.env.reset()
        total_reward = 0
        for _ in range(1000):
            action = self.get_action()
            next_state, reward, done, _ = self.env.step(action=action)
            total_reward += reward
            my_agent.put_step_in_trajectory(action=action, 
                                            state=self.get_state(next_state)
                                            )
            
            if visualize:
                time.sleep(0.1)
                self.env.render()

            if done:
                break
        
        self.list_trajectories["trajectory"].append(self.trajectory)
        self.list_trajectories["reward"].append(reward)


    def make_sample_trajectories(self, num_samples=20, visualize=False):
        for _ in range(num_samples):
            self.sample_one_tajectory(visualize)
        return None

    def policy_evaluation(self, num_samples=20):
        self.make_sample_trajectories(num_samples=num_samples)
        return np.mean(self.list_trajectories["reward"])






action_n = 4

my_agent = RandomAgent(action_n)
print(f"{my_agent.policy_evaluation(num_samples=20)=}")
# env = gym.make('maze-sample-5x5-v0')
# observation, info = env.reset()
# print(observation)
# print(info)
# total_reward = 0

# for _ in range(1000):
#     action = my_agent.get_action()
#     next_state, reward, done, _ = env.step(action=action)
#     total_reward += reward
#     my_agent.put_step(action=action, state=my_agent.get_state(next_state))
#     time.sleep(0.1)
#     env.render()

#     if done:
#         break

# print(f"{total_reward=}")
# print(my_agent.trajectory)