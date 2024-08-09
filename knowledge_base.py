import numpy as np
from scipy.spatial import cKDTree
from collections import deque

class KnowledgeBase:
    def __init__(self, max_size=1000):
        self.experiences = deque(maxlen=max_size)

    def add_experience(self, experience):
        self.experiences.appendleft(experience)

    def get_relevant_experience(self, goal_space, goal):
        if not self.experiences:
            return None
        
        test_scores = []
        for i in range(len(self.experiences)):
            test_scores.append((i, goal_space.get_fitness(self.experiences[i].final_observation, goal)))

        #if goal_space.name == 'agent':
        #    print(f'\n--------LIST OF FITNESS SCORES FOR CURRENT AGENT GOAL--------\nExperiences Deque is: {len(self.experiences)} long.\n{test_scores}')

        best_experience = max(self.experiences, 
                              key=lambda exp: goal_space.get_fitness(exp.final_observation, goal))
        
        #if goal_space.name == 'agent':
        #    print(f'INDEX OF BEST PREVIOUS AGENT EXPERIENCE:{self.experiences.index(best_experience)}\n')

        return best_experience