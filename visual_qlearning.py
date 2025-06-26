import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from time import sleep
from IPython.display import clear_output  # If using Jupyter Notebook

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def main(learning_rate=0.9, discount_rate=0.8, num_episodes=1000):
    # Create Taxi environment
    env = gym.make('Taxi-v3', render_mode='ansi')  # Changed to 'human' for visualization
    
    # Initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # Hyperparameters
    epsilon = 1.0
    decay_rate = 0.005
    
    # Training variables
    max_steps = 99  # per episode
    rewards = []
    epsilons = []
    steps_per_episode = []
    
    # Create figure for live plotting
    plt.figure(figsize=(12, 8))
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        step = 0
        
        for s in range(max_steps):
            # Exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(qtable[state, :])  # Exploit
            
            # Take action and observe reward
            new_state, reward, done, info, _ = env.step(action)
            total_reward += reward
            
            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + learning_rate * (
                reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]
            )
            
            # Update state
            state = new_state
            step += 1
            
            if done:
                break
        
        # Store metrics for visualization
        rewards.append(total_reward)
        epsilons.append(epsilon)
        steps_per_episode.append(step)
        
        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)
        
        # Print training progress
        if episode % 100 == 0 or episode == num_episodes - 1:
            print(f"Episode: {episode}, Reward: {total_reward}, Steps: {step}, Epsilon: {epsilon:.2f}")
            
        # Update plots every 100 episodes
        if episode % 100 == 0 or episode == num_episodes - 1:
            clear_output(wait=True)  # For Jupyter Notebook
            update_plots(rewards, epsilons, steps_per_episode, episode, num_episodes)
    
    # Final evaluation
    evaluate_agent(env, qtable, max_steps)
    
    # Save Q-table and plots
    np.save('qtable.npy', qtable)
    save_final_plots(rewards, epsilons, steps_per_episode)
    
    env.close()

def update_plots(rewards, epsilons, steps, current_episode, total_episodes):
    """Update live training plots"""
    plt.clf()
    
    # Reward plot
    plt.subplot(3, 1, 1)
    plt.plot(rewards, 'b')
    plt.title(f'Training Progress (Episode {current_episode}/{total_episodes})')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Epsilon plot
    plt.subplot(3, 1, 2)
    plt.plot(epsilons, 'r')
    plt.ylabel('Exploration Rate (Epsilon)')
    plt.grid(True)
    
    # Steps plot
    plt.subplot(3, 1, 3)
    plt.plot(steps, 'g')
    plt.ylabel('Steps per Episode')
    plt.xlabel('Episodes')
    plt.grid(True)
    
    plt.tight_layout()
    plt.pause(0.001)  # Pause to update the plot
    plt.show()

import gymnasium as gym

env = gym.make("Taxi-v3")
state = env.reset()[0]  # Get initial state

# Named locations
locations = ['R', 'G', 'Y', 'B']

def decode_taxi_state(state):
    out = {}

    taxi_row = state // 100
    taxi_col = (state % 100) // 20
    passenger_location = (state % 20) // 4
    destination = state % 4

    out["taxi_row"] = taxi_row
    out["taxi_col"] = taxi_col
    out["passenger"] = locations[passenger_location] if passenger_location < 4 else "IN_TAXI"
    out["destination"] = locations[destination]
    return out


def evaluate_agent(env, qtable, max_steps, num_episodes=10):
    """Evaluate the trained agent"""
    print("\nEvaluating trained agent...")
    total_rewards = []

    action_meanings = {
        0: "South",
        1: "North",
        2: "East",
        3: "West",
        4: "Pickup",
        5: "Dropoff"
    }
    
    for episode in range(num_episodes):
        state = env.reset()[0]

        print(f"\nEvaluation Episode {episode + 1}")
        print("-"*25)
        print(env.render())

        decoded = decode_taxi_state(state)
        print(f"Taxi at ({decoded['taxi_row']}, {decoded['taxi_col']})")
        print(f"Passenger at {decoded['passenger']}")
        print(f"Destination is {decoded['destination']}")

        done = False
        total_reward = 0
        
        for s in range(max_steps):
            action = np.argmax(qtable[state, :])
            new_state, reward, done, info, _ = env.step(action)
            total_reward += reward
            
            # Render the environment (visualization)
            #env.render()
            print(f"Step: {s + 1}, Action: {action_meanings[action]}, Reward: {reward}, Total Reward: {total_reward}")
            sleep(0.5)  # Slow down for visualization
            
            state = new_state
            if done:
                break
        
        total_rewards.append(total_reward)
        print(f"Episode finished with total reward: {total_reward}")
    
    print(f"\nAverage reward over {num_episodes} evaluation episodes: {np.mean(total_rewards):.2f}")

def save_final_plots(rewards, epsilons, steps, output_name='training_results.png'):
    """Save final training plots"""
    plt.figure(figsize=(12, 8))
    
    # Final rewards plot
    plt.subplot(3, 1, 1)
    plt.plot(rewards, 'b')
    plt.title('Training Results')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Final epsilon plot
    plt.subplot(3, 1, 2)
    plt.plot(epsilons, 'r')
    plt.ylabel('Exploration Rate (Epsilon)')
    plt.grid(True)
    
    # Final steps plot
    plt.subplot(3, 1, 3)
    plt.plot(steps, 'g')
    plt.ylabel('Steps per Episode')
    plt.xlabel('Episodes')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close()

if __name__ == "__main__":
    main()