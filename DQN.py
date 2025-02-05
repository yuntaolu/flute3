# version 3
# load ckpt
# checkpoint = torch.load('results/checkpoint_episode_100.pth')
# policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# episode_durations = checkpoint['episode_durations']
# episode_rewards = checkpoint['episode_rewards']
# losses = checkpoint['losses']
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import time
import os
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 创建保存结果的目录
if not os.path.exists('results'):
    os.makedirs('results')

# 设置随机种子
random.seed(0)
torch.manual_seed(0)

# 创建环境
env = gym.make('CartPole-v0')

# 设置matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cpu")

Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 训练参数
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = []
episode_rewards = []
losses = []

def plot_training(show_result=False):
    plt.figure(figsize=(15, 5))
    
    # Plot durations
    plt.subplot(131)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Final Result')
    else:
        plt.title('Training Duration')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), 'r-')
    
    # Plot rewards
    plt.subplot(132)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), 'r-')
    
    # Plot losses
    if losses:
        plt.subplot(133)
        losses_t = torch.tensor(losses, dtype=torch.float)
        plt.title('Training Loss')
        plt.xlabel('Training steps')
        plt.ylabel('Loss')
        plt.plot(losses_t.numpy())

    plt.tight_layout()
    
    if show_result:
        plt.savefig('results/final_training_results.png')
    else:
        plt.savefig('results/training_progress.png')
    
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                     if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    return loss.item()

def format_time(seconds):
    """格式化时间"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f'{h:d}h{m:02d}m{s:02d}s'
    elif m > 0:
        return f'{m:02d}m{s:02d}s'
    else:
        return f'{s:02d}s'

def train():
    num_episodes = 50
    print_interval = 10
    start_time = time.time()
    
    # 创建总进度条
    with trange(num_episodes, desc="Training Progress") as t:
        for i_episode in t:
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0
            steps = 0
            
            # 创建每个episode的进度条
            with tqdm(desc=f"Episode {i_episode+1}", leave=False) as episode_pbar:
                for step in count():
                    action = select_action(state)
                    observation, reward, done, _ = env.step(action.item())
                    episode_reward += reward
                    reward = torch.tensor([reward], device=device)

                    if done:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                    memory.push(state, action, next_state, reward)
                    state = next_state

                    loss = optimize_model()
                    if loss is not None:
                        losses.append(loss)

                    # 更新目标网络
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    target_net.load_state_dict(target_net_state_dict)

                    steps += 1
                    episode_pbar.update(1)
                    episode_pbar.set_description(f"Episode {i_episode+1} - Steps: {steps}")

                    if done:
                        episode_durations.append(steps)
                        episode_rewards.append(episode_reward)
                        plot_training()
                        break

            # 更新总进度条信息
            elapsed_time = time.time() - start_time
            avg_reward = sum(episode_rewards[-print_interval:]) / min(len(episode_rewards), print_interval)
            avg_duration = sum(episode_durations[-print_interval:]) / min(len(episode_durations), print_interval)
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            
            t.set_postfix({
                'Time': format_time(elapsed_time),
                'Reward': f'{avg_reward:.2f}',
                'Duration': f'{avg_duration:.2f}',
                'Epsilon': f'{epsilon:.2f}'
            })

            # 每隔一定间隔保存checkpoint
            if (i_episode + 1) % 50 == 0:
                torch.save({
                    'episode': i_episode,
                    'policy_net_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_durations': episode_durations,
                    'episode_rewards': episode_rewards,
                    'losses': losses
                }, f'results/checkpoint_episode_{i_episode+1}.pth')

    print('\nTraining Complete')
    plot_training(show_result=True)
    plt.ioff()
    
    # 保存最终模型
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_durations': episode_durations,
        'episode_rewards': episode_rewards,
        'losses': losses
    }, 'results/dqn_model_final.pth')
    
    return policy_net

def test_model(model, num_episodes=10):
    """测试训练好的模型"""
    print("\nTesting Model...")
    model.eval()
    
    test_rewards = []
    with torch.no_grad():
        for i in trange(num_episodes, desc="Testing Progress"):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0
            done = False
            
            while not done:
                action = model(state).max(1)[1].view(1, 1)
                observation, reward, done, _ = env.step(action.item())
                episode_reward += reward
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            test_rewards.append(episode_reward)
            
    avg_reward = sum(test_rewards) / num_episodes
    print(f"\nAverage Test Reward: {avg_reward:.2f}")
    return test_rewards

if __name__ == "__main__":
    print("Starting DQN Training on CartPole-v0")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Episodes: {500}")
    print("=" * 50)
    
    try:
        trained_model = train()
        test_model(trained_model)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        env.close()

'''
# version 2
# load model
# checkpoint = torch.load('results/dqn_model.pth')
# policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 创建保存结果的目录
if not os.path.exists('results'):
    os.makedirs('results')

# 设置随机种子
random.seed(0)
torch.manual_seed(0)

# 创建环境
env = gym.make('CartPole-v0')

# 设置matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cpu")

Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 训练参数
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = []
episode_rewards = []
losses = []

def plot_training(show_result=False):
    plt.figure(figsize=(15, 5))
    
    # Plot durations
    plt.subplot(131)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Final Result')
    else:
        plt.title('Training Duration')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), 'r-')
    
    # Plot rewards
    plt.subplot(132)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), 'r-')
    
    # Plot losses
    if losses:
        plt.subplot(133)
        losses_t = torch.tensor(losses, dtype=torch.float)
        plt.title('Training Loss')
        plt.xlabel('Training steps')
        plt.ylabel('Loss')
        plt.plot(losses_t.numpy())

    plt.tight_layout()
    
    if show_result:
        plt.savefig('results/final_training_results.png')
    else:
        plt.savefig('results/training_progress.png')
    
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                     if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    return loss.item()

def train():
    num_episodes = 500
    print_interval = 10
    start_time = time.time()
    
    for i_episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0
        
        for t in count():
            action = select_action(state)
            observation, reward, done, _ = env.step(action.item())
            episode_reward += reward
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state

            loss = optimize_model()
            if loss is not None:
                losses.append(loss)

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(episode_reward)
                plot_training()
                break
        
        # 打印训练进度
        if (i_episode + 1) % print_interval == 0:
            avg_reward = sum(episode_rewards[-print_interval:]) / print_interval
            avg_duration = sum(episode_durations[-print_interval:]) / print_interval
            elapsed_time = time.time() - start_time
            print(f'Episode {i_episode+1}/{num_episodes} | '
                  f'Avg Reward: {avg_reward:.2f} | '
                  f'Avg Duration: {avg_duration:.2f} | '
                  f'Epsilon: {EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY):.2f} | '
                  f'Time: {elapsed_time:.1f}s')

    print('Training Complete')
    plot_training(show_result=True)
    plt.ioff()
    
    # 保存模型
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_durations': episode_durations,
        'episode_rewards': episode_rewards,
        'losses': losses
    }, 'results/dqn_model.pth')
    
    return policy_net

if __name__ == "__main__":
    trained_model = train()
    env.close()
'''
    
# version 1
'''
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 设置随机种子
random.seed(0)
torch.manual_seed(0)

# 创建环境
env = gym.make('CartPole-v0')

# 设置matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# 设置设备
device = torch.device("cpu")

# 定义经验元组
Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))

# 定义回放内存
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 训练参数
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# 获取环境参数
n_actions = env.action_space.n
state = env.reset()  # 修改这里，直接获取状态
n_observations = len(state)

# 初始化网络
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

# 选择动作函数
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# 绘图函数
episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# 训练函数
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                     if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# 主训练循环
def train():
    num_episodes = 500
    for i_episode in range(num_episodes):
        state = env.reset()  # 修改这里，直接获取状态
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in count():
            action = select_action(state)
            observation, reward, done, _ = env.step(action.item())  # 修改这里，适配gym 0.17.3
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()
'''

'''
# Copy from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
'''