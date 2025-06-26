import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

from visual_qlearning import save_final_plots

class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Learning Agent"""
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.997, 
                 epsilon=1, epsilon_decay=0.99, epsilon_min=0.01,
                 buffer_size=25000, batch_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_size)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.choice(np.arange(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save the complete agent state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size
            }
        }, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load the complete agent state"""
        checkpoint = torch.load(filepath)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore hyperparameters
        params = checkpoint['hyperparameters']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_min = params['epsilon_min']
        
        print(f"Agent loaded from {filepath}")
    
    @staticmethod
    def load_agent(filepath):
        """Create and load a complete agent from file"""
        checkpoint = torch.load(filepath)
        params = checkpoint['hyperparameters']
        
        # Create agent with saved hyperparameters
        agent = DQNAgent(
            state_size=params['state_size'],
            action_size=params['action_size'],
            lr=params['lr'],
            gamma=params['gamma'],
            epsilon=params['epsilon'],
            epsilon_decay=params['epsilon_decay'],
            epsilon_min=params['epsilon_min'],
            batch_size=params['batch_size']
        )
        
        # Load the trained weights
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return agent

def state_to_features(state):
    """Convert discrete state to feature vector"""
    # Taxi environment has 500 discrete states
    # We'll use one-hot encoding
    features = np.zeros(500)
    features[state] = 1.0
    return features

def train_dqn(lr=0.001, gamma=0.997, 
                 epsilon=1, epsilon_decay=0.99, epsilon_min=0.01,
                batch_size=128, episodes=20000):
    """Train DQN agent on Taxi environment"""
    env = gym.make('Taxi-v3')
    state_size = 500  # One-hot encoded state
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, lr, gamma, 
                 epsilon, epsilon_decay, epsilon_min,
                 batch_size)
    
    target_update_freq = 100
    
    scores = []
    avg_scores = []
    epsilons = []
    each_step = []
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle new gym API
        
        state = state_to_features(state)
        total_reward = 0
        done = False
        
        while not done:
            tmp = 0
            action = agent.act(state)
            result = env.step(action)
            
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            
            next_state = state_to_features(next_state)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break

            tmp += 1

        each_step.append(tmp)
        epsilons.append(agent.epsilon)

        # Train the agent
        agent.replay()
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    save_final_plots(scores, epsilons, each_step, output_name='new_deep_q_results.png')

    return agent, scores, avg_scores

def test_agent(agent, episodes=10):
    """Test the trained agent"""
    env = gym.make('Taxi-v3')
    agent.epsilon = 0  # No exploration during testing
    
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        state = state_to_features(state)
        total_reward = 0
        done = False
        
        while not done:
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
        
        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Reward = {total_reward}")
    
    print(f"Average Test Reward: {np.mean(total_rewards):.2f}")
    return total_rewards

def plot_results(scores, avg_scores):
    """Plot training results and return the figure"""
    fig = plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_scores)
    plt.title('Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    return fig
    

if __name__ == "__main__":
    print("Training DQN Agent...")
    agent, scores, avg_scores = train_dqn()

    agent.save('taxi_dqn_agent.pth')
    
    print("\nTesting trained agent...")
    test_rewards = test_agent(agent)
    
    print("\nPlotting results...")
    # fig = plot_results(scores, avg_scores)
    # 