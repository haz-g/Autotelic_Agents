from env import MinecraftCartEnv
from goal_spaces import GoalSpaceManager
from knowledge_base import KnowledgeBase
from policy_manager import PolicyManager
from neural_network import NeuralNetwork
from experience import Experience
import random
import numpy as np
from utils import action_to_string

NUM_ITERATIONS = 40000
EXPLORE_PROB = 0.8

def main():
    # Initialize components
    env = MinecraftCartEnv()
    goal_space_manager = GoalSpaceManager()
    knowledge_base = KnowledgeBase()
    neural_network = NeuralNetwork(input_dim=18, hidden_dim=64, output_dim=5)  # 18 for observation + 2 for goal, 5 possible actions
    policy_manager = PolicyManager(neural_network)

    # Variables for tracking progress:
    exploration_count = 0
    exploitation_count = 0
    agent_exploitations = 0
    pickaxe_exploitations = 0
    shovel_exploitations = 0
    cart_exploitations = 0
    block_exploitations = 0
    agent_current_LP = 0
    pickaxe_current_LP = 0
    shovel_current_LP = 0
    cart_current_LP = 0
    block_current_LP = 0

    for iteration in range(NUM_ITERATIONS):
        # Choose goal space and goal
        goal_space = goal_space_manager.choose_goal_space()
        goal = goal_space.sample_goal()

        # Choose exploration or exploitation [Plus record amount of choice for info later]
        policy_pick = round(random.random(), 1)
        if policy_pick <= EXPLORE_PROB:
            policy_type = 'explore'
            exploration_count += 1
        else:
            policy_type = 'exploit'
            exploitation_count += 1

        if  policy_type == 'explore':
            policy = policy_manager.exploration_policy
        elif policy_type == 'exploit':
            policy = policy_manager.exploitation_policy

        # Select relevant experience
        relevant_experience = knowledge_base.get_relevant_experience(goal_space, goal)

        # Execute policy
        trajectory, final_observation = policy.execute(env, goal_space, goal, relevant_experience)

        # Create new experience
        new_experience = Experience(goal_space, goal, trajectory, final_observation)

        # Update knowledge base
        knowledge_base.add_experience(new_experience)

        # Update goal space (if exploiting)
        if policy == policy_manager.exploitation_policy:
            goal_space.update_learning_progress(new_experience)
            if goal_space.name == 'agent':
                agent_exploitations += 1
                print(f'\n-----AGENT LP UPDATE-----\nNEW LP: {goal_space.learning_progress}\nNEW GOAL DATA:{[goal_space.goal_data[i]['learning_progress'] for i in goal_space.goal_data]}\nAGENT PATH:{[action_to_string(i[18:21]) for i in new_experience.trajectory]}\n')
                agent_current_LP = round(goal_space.learning_progress,3)
            elif goal_space.name == 'pickaxe':
                pickaxe_exploitations += 1
                pickaxe_current_LP = round(goal_space.learning_progress,3)
            elif goal_space.name == 'shovel':
                shovel_exploitations += 1
                shovel_current_LP = round(goal_space.learning_progress,3)
            elif goal_space.name == 'cart':
                cart_exploitations += 1
                cart_current_LP = round(goal_space.learning_progress,3)
            elif goal_space.name == 'blocks':
                block_exploitations += 1
                block_current_LP = round(goal_space.learning_progress,3)
            
        # Print progress
        if iteration % 10 == 0:
            print(f"-------Iteration {iteration}-------\nAGENT EXPLOITS: {agent_exploitations} | CURRENT LP: {agent_current_LP}\nPICKAXE EXPLOITS: {pickaxe_exploitations} | CURRENT LP: {pickaxe_current_LP}\nSHOVEL EXPLOITS: {shovel_exploitations} | CURRENT LP: {shovel_current_LP}\nCART EXPLOITS: {cart_exploitations} | CURRENT LP: {cart_current_LP}\nBLOCKS EXPLOITS: {block_exploitations} | CURRENT LP: {block_current_LP}\n")

        if iteration % 100 == 0:
            overall_progress = np.mean([gs.learning_progress for gs in goal_space_manager.goal_spaces.values()])
            print(f"\n\n-------OVERALL LP: {overall_progress:.4f}--------\nFrom {exploitation_count} exploitations and {exploration_count} explorations\nNumber of policies in parameter space: {policy_manager.current_key}\n------------------------------\n\n")

    # After all iterations, print final statistics
    print("\nFinal Statistics:")
    for name, goal_space in goal_space_manager.goal_spaces.items():
        print(f"{name} - Learning Progress: {goal_space.learning_progress:.4f}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()