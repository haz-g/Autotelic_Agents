import numpy as np

class Experience:
    def __init__(self, goal_space, goal, trajectory, final_observation):
        self.goal_space = goal_space
        self.goal = goal
        self.trajectory = trajectory
        self.final_observation = final_observation
        self.fitness = self._calculate_fitness()

    def _calculate_fitness(self):
        return self.goal_space.get_fitness(self.final_observation, self.goal)

    def get_relevant_trajectory(self, current_goal_space):
        relevant_trajectory = []
        for step in self.trajectory:
            relevant_obs = current_goal_space.get_relevant_observation(step[:18])
            relevant_trajectory.append(relevant_obs + step[18:])
        return relevant_trajectory

    def __repr__(self):
        return f"Experience(goal_space={self.goal_space.name}, goal={self.goal}, fitness={self.fitness:.4f})"