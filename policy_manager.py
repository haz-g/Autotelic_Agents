import numpy as np
import torch
import random

class PolicyManager:
    def __init__(self, neural_network):
        self.nn = neural_network
        self.parameter_space = {1: self.nn.get_parameters()}
        self.current_key = 1
        self.exploration_policy = ExplorationPolicy(self)
        self.exploitation_policy = ExploitationPolicy(self)
        self.last_action = None  # To keep track of the last action
        self.repeated_tool_use_count = 0

    def select_action(self, observation):
        with torch.no_grad():
            input_tensor = torch.tensor(observation, dtype=torch.float32)
            action_probs = self.nn(input_tensor)
            
            if self.last_action is not None and np.array_equal(self.last_action, [0, 0, 1]):
                self.repeated_tool_use_count += 1 
            
            # If the last two actions were "use tool", exclude it from possible actions
            if self.repeated_tool_use_count == 2:
                self.repeated_tool_use_count = 0
                action_probs[4] = 0  # Set probability of "use tool" to 0
                if torch.sum(action_probs) == 0:
                    # If all probabilities are zero, choose randomly from other actions
                    action_index = random.randint(0, 3)
                else:
                    action_probs = action_probs / torch.sum(action_probs)  # Renormalize
                    action_index = torch.multinomial(action_probs, 1).item()
            else:
                action_index = torch.multinomial(action_probs, 1).item()

        # Convert action index to the required format
        if action_index == 0:  # forward
            action = np.array([-1, 0, 0])
        elif action_index == 1:  # backwards
            action = np.array([1, 0, 0])
        elif action_index == 2:  # left
            action = np.array([0, -1, 0])
        elif action_index == 3:  # right
            action = np.array([0, 1, 0])
        else:  # use
            action = np.array([0, 0, 1])

        self.last_action = action  # Update the last action
        return action

    def mutate_parameters(self, start_index, relevant_experience):
        trajectory = relevant_experience.trajectory
        param_list = []
        
        # Add original parameter keys up to start_index
        for step in trajectory[:start_index]:
            param_list.append(step[-1])
        
        # Get the parameters at start_index and mutate them
        original_params = self.parameter_space[trajectory[start_index][-1]]
        mutated_params = {name: param.clone() for name, param in original_params.items()}
        for param in mutated_params.values():
            noise = torch.randn_like(param) * 0.3
            param.add_(noise)
        
        # Add the new mutated parameters to the parameter space with a new key
        self.current_key += 1
        new_key = self.current_key
        self.parameter_space[new_key] = mutated_params
        
        # Fill the rest of param_list with the new key
        param_list.extend([new_key] * (40 - len(param_list)))
        assert len(param_list) == 40, f"param_list should be length 40, but is {len(param_list)}"
        
        return param_list

class ExplorationPolicy:
    def __init__(self, policy_manager):
        self.policy_manager = policy_manager

    def execute(self, env, goal_space, goal, relevant_experience):
        self.repeated_tool_use_count = 0
        self.policy_manager.last_action = None
        trajectory = []
        observation = env.observation()
        
        if relevant_experience is not None:
            mutation_start = self._find_mutation_start(relevant_experience, goal_space, goal)
            param_keys = self.policy_manager.mutate_parameters(mutation_start, relevant_experience)
            
            # Use previous actions up to mutation_start
            for i in range(mutation_start):
                action = relevant_experience.trajectory[i][18:21]  # Extract action from previous experience
                next_observation, done = env.step(action[0], action[1], action[2])
                trajectory.append(tuple(observation) + tuple(action) + (param_keys[i],))
                observation = next_observation
                if done:
                    return trajectory, observation
            
            # Use mutated parameters from mutation_start onwards
            for a in range(mutation_start, 40):
                self.policy_manager.nn.set_parameters(self.policy_manager.parameter_space[param_keys[a]])
                action = self.policy_manager.select_action(observation)
                next_observation, done = env.step(action[0], action[1], action[2])
                trajectory.append(tuple(observation) + tuple(action) + (param_keys[a],))
                observation = next_observation
                if done:
                    break
        else:
            # If no relevant experience, use initial parameters for all steps
            for a in range(40):
                action = self.policy_manager.select_action(observation)
                next_observation, done = env.step(action[0], action[1], action[2])
                trajectory.append(tuple(observation) + tuple(action) + (1,))
                observation = next_observation
                if done:
                    break

        return trajectory, observation

    def _find_mutation_start(self, experience, current_goal_space, current_goal):
        best_fitness = 0
        mutation_start = 0
        for i, step in enumerate(experience.trajectory):
            fitness = current_goal_space.get_fitness(step[:18], current_goal)
            if fitness > best_fitness:
                best_fitness = fitness
                mutation_start = i + 1  # Start mutating from the next step
            elif fitness < best_fitness:
                break  # Stop when fitness starts decreasing
        return mutation_start

class ExploitationPolicy:
    def __init__(self, policy_manager):
        self.policy_manager = policy_manager
        self.exploration_rate = 0.1  # Small chance to explore during exploitation

    def execute(self, env, goal_space, goal, relevant_experience):
        self.repeated_tool_use_count = 0
        self.policy_manager.last_action = None
        trajectory = []
        observation = env.observation()
        
        if relevant_experience is not None:
            for step in relevant_experience.trajectory:
                if random.random() < self.exploration_rate:
                    # Small chance to explore
                    self.policy_manager.nn.set_parameters(self.policy_manager.parameter_space[step[-1]])
                    action = self.policy_manager.select_action(observation)
                else:
                    # Otherwise, use the action from the best trajectory
                    action = step[18:21]
                
                next_observation, done = env.step(action[0], action[1], action[2])
                trajectory.append(tuple(observation) + tuple(action) + (step[-1],))
                observation = next_observation
                if done:
                    break
        else:
            # If just run on current params (should only be for when exploit is picked first)
            for _ in range(40):
                action = self.policy_manager.select_action(observation)
                next_observation, done = env.step(action[0], action[1], action[2])
                trajectory.append(tuple(observation) + tuple(action) + (1,))  # Assuming 1 is the key for initial parameters
                observation = next_observation
                if done:
                    break

        return trajectory, observation