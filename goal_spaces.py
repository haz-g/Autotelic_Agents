import numpy as np
import random

class GoalSpace:
    def __init__(self, name, dimension):
        self.name = name
        self.dimension = dimension
        self.goals = self._initialize_goals()
        self.learning_progress = 0
        self.goal_data = {}
        self.decay_factor = 0.95  # Decay factor for stagnant progress
        self.cumulative_progress = 0  # Cumulative progress measure

    def _initialize_goals(self):
        if self.name == 'agent':
            return {
                        0: [0.471, 0.261], # Position behind diamond blocks
                        1: [0.471, 0.435], # Position Behind one of the walls
                        2: [0.412, 0.565], # Where the Pickaxe is
                        3: [0.529, 0.565], # Where the Spade is
                    }
        elif self.name in ['pickaxe', 'shovel']:
            return {
                        0: [0.471, 0.261], # Position behind diamond blocks
                        1: [0.471, 0.435], # Position Behind one of the walls
                        2: [0.235, 0.217], # On back track left
                        3: [0.706, 0.217], # On back track right
                    }
        elif self.name == 'cart':
            return {
                        0: 0.176, # On track far left
                        1: 0.765, # On track far right
                        2: 0.235, # On track medium left pos
                        3: 0.706, # On track medium right pos
                    }
        elif self.name == 'blocks':
            return {
                        0: [0, 0, 1, 1, 1],
                        1: [1, 1, 0, 1, 1],
                        2: [1, 1, 1, 0, 0],
                    }
        else:
            raise ValueError(f"Unknown goal space: {self.name}")

    def sample_goal(self):
        return random.choice(list(self.goals.values()))

    def update_learning_progress(self, new_experience):
        goal = str(new_experience.goal)
        new_fitness = new_experience.fitness
        
        if goal in self.goal_data:
            old_fitness = self.goal_data[goal]['last_fitness']
            immediate_progress = new_fitness - old_fitness
            
            # Update cumulative progress
            if immediate_progress > 0:
                self.cumulative_progress += immediate_progress
            elif immediate_progress < 0:
                self.cumulative_progress = max(0, self.cumulative_progress + immediate_progress)
            
            # Calculate learning progress
            if immediate_progress > 0:
                learning_progress = immediate_progress + 0.1 * self.cumulative_progress
            elif immediate_progress < 0:
                learning_progress = immediate_progress - 0.1 * self.cumulative_progress
            else:
                learning_progress = self.goal_data[goal]['learning_progress'] * self.decay_factor
            
            self.goal_data[goal]['last_fitness'] = new_fitness
            self.goal_data[goal]['learning_progress'] = learning_progress
        else:
            self.goal_data[goal] = {'last_fitness': new_fitness, 'learning_progress': new_fitness}
        
        # Update overall learning progress for this goal space
        active_goals = [data['learning_progress'] for data in self.goal_data.values() if data['learning_progress'] != 0]
        self.learning_progress = np.mean(active_goals) if active_goals else 0

    def get_fitness(self, observation, goal):
        observation = np.array(observation)
        goal = np.array(goal)

        if self.name in ['agent', 'pickaxe', 'shovel']:
            return self._get_position_fitness(observation, goal)
        elif self.name == 'cart':
            return self._get_cart_fitness(observation, goal)
        elif self.name == 'blocks':
            return self._get_blocks_fitness(observation, goal)
        else:
            raise ValueError(f"Unknown goal space: {self.name}")

    def _get_position_fitness(self, observation, goal):
        if self.name == 'agent':
            pos = observation[:2]
            if pos[1] > 0.565:  # In the lower half of the map
                return self._lower_half_fitness(pos)
            else:  # In the upper half of the map
                return self._upper_half_fitness(pos, goal)
        elif self.name == 'pickaxe':
            pos = observation[2:4]
            if round(pos[0],3)-0.412 <= 0.01 and round(pos[1],3)-0.565 <= 0.01:  # Pickaxe not collected
                return 0
        elif self.name == 'shovel':
            pos = observation[4:6]
            if round(pos[0],3)-0.529 <= 0.01 and round(pos[1],3)-0.565 <= 0.01:
                return 0

        # For pickaxe and shovel when collected, or for other cases
        return self._goal_directed_fitness(pos, goal)

    def _lower_half_fitness(self, pos):
        path_positions = [
            (0.471, 0.739, 0.05),  # centre tile
            (0.471, 0.696, 0.05), (0.471, 0.652, 0.045), (0.471, 0.609, 0.040), # path in front of centre
            (0.412, 0.739, 0.075), (0.529, 0.739, 0.075),  # first left and right
            (0.353, 0.739, 0.10), (0.588, 0.739, 0.10),  # second left and right
            (0.353, 0.696, 0.20), (0.588, 0.696, 0.20),  # third left and right
            (0.353, 0.652, 0.25), (0.588, 0.652, 0.25),  # fourth left and right
            (0.294, 0.652, 0.30), (0.647, 0.652, 0.30),  # fifth left and right
            (0.294, 0.609, 0.35), (0.647, 0.609, 0.35),  # sixth left and right
        ]
            
        for x, y, reward in path_positions:
            if np.allclose(pos, [x, y], atol=0.01):
                return reward
        
        return 0  # If not on the path

    def _upper_half_fitness(self, pos, goal):
        base_fitness = self._goal_directed_fitness(pos, goal)
        lower_half_max = 0.35  # Maximum reward for lower half
        
        if base_fitness <= lower_half_max:
            return lower_half_max + 0.15  # Boost it above lower half max
        else:
            return base_fitness

    def _goal_directed_fitness(self, pos, goal):
        y_diff = abs(pos[1] - goal[1])
        x_diff = abs(pos[0] - goal[0])
        
        y_fitness = np.exp(-5 * y_diff)  # Exponential weighting for y-coordinate
        
        if y_diff < 0.05:  # If close enough to correct y-coordinate
            x_fitness = np.exp(-5 * x_diff)  # Exponential weighting for x-coordinate
            return min((y_fitness + x_fitness) / 2, 1)  # Ensure it doesn't exceed 1
        else:
            return min(y_fitness / 2, 1)  # Ensure it doesn't exceed 1

    def _get_cart_fitness(self, observation, goal):
        cart_pos = observation[6]
        if abs(cart_pos - 0.471) < 0.01:  # Cart in starting position
            return 0
        
        distance = abs(cart_pos - goal)
        return np.exp(-5 * distance)  # Exponential weighting based on distance

    def _get_blocks_fitness(self, observation, goal):
        blocks_state = observation[11:16]
        target_blocks = np.where(goal == 0)[0]
        blocks_to_break = len(target_blocks)
        
        if blocks_to_break == 0:
            return 1  # All blocks already in desired state
        
        broken_correct_blocks = sum(blocks_state[i] == 0 for i in target_blocks)
        
        if broken_correct_blocks == 0:
            return 0  # No correct blocks broken
        
        return np.exp(-2 * (blocks_to_break - broken_correct_blocks))  # Exponential weighting

class GoalSpaceManager:
    def __init__(self):
        self.goal_spaces = {
            'agent': GoalSpace('agent', 2),
            'pickaxe': GoalSpace('pickaxe', 2),
            'shovel': GoalSpace('shovel', 2),
            'cart': GoalSpace('cart', 1),
            'blocks': GoalSpace('blocks', 5)
        }

    def choose_goal_space(self):
        if round(random.random(), 1) <= 0.8:  # 80% chance of choosing based on learning progress
            lp_values = [gs.learning_progress for gs in self.goal_spaces.values()]
            total_lp = sum(lp_values)
            if total_lp == 0:
                return random.choice(list(self.goal_spaces.values()))
            else:
                chosen_index = np.random.choice(len(self.goal_spaces), p=[lp/total_lp for lp in lp_values])
                return list(self.goal_spaces.values())[chosen_index]
        else:  # 20% chance of random selection
            return random.choice(list(self.goal_spaces.values()))