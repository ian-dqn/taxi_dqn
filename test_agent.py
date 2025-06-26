import gym
import numpy as np

from deep_qlearning import DQNAgent, state_to_features

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def load_agent(filepath):
    return DQNAgent.load_agent(filepath)

def evaluate_agent(agent, episodes=10, render=False):
    try:
        agent.epsilon = 0  # No exploration during testing
        
        env = gym.make('Taxi-v3', render_mode='human' if render else None)
        
        total_rewards = []
        successful_episodes = 0
        total_steps = []
        
        print(f"Testing agent for {episodes} episodes...")
        print("-" * 50)
        
        for episode in range(episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            state = state_to_features(state)
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 200:  # Prevent infinite episodes
                action = agent.act(state)
                result = env.step(action)
                
                if len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                
                next_state = state_to_features(next_state)
                state = next_state
                total_reward += reward
                steps += 1
                
                if render:
                    env.render()
            
            total_rewards.append(total_reward)
            total_steps.append(steps)
            
            # Check if episode was successful (reward = 20 means successful pickup and dropoff)
            if total_reward > 0:
                successful_episodes += 1
                status = "SUCCESS"
            else:
                status = "FAILED"
            
            print(f"Episode {episode + 1:2d}: Reward = {total_reward:3d}, Steps = {steps:3d} {status}")
        
        env.close()
        
        # Calculate statistics
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(total_steps)
        success_rate = successful_episodes / episodes
        
        print("-" * 50)
        print(f"Test Results Summary:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps:  {avg_steps:.1f}")
        print(f"Success Rate:   {success_rate:.2%} ({successful_episodes}/{episodes})")
        print(f"Best Episode:   {max(total_rewards)}")
        print(f"Worst Episode:  {min(total_rewards)}")
        
        return {
            'rewards': total_rewards,
            'steps': total_steps,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'success_rate': success_rate,
            'successful_episodes': successful_episodes
        }
        
    except FileNotFoundError:
        print(f"Error: Agent file '{filepath}' not found!")
        return None
    except Exception as e:
        print(f"Error loading agent: {e}")
        return None


if __name__ == "__main__":
    print("Test Agent...")
    
    filepath = r'./dqn_agents/dqn_taxi.pth'

    agent = load_agent(filepath)
    evaluate_agent(agent, episodes=1000, render=False)