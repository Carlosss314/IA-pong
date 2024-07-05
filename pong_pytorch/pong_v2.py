import gym
import torch
import torch.nn as nn
from collections import deque

render = True
resume = True

def prepro(I):
  I = I[35:195]
  I = I[::2,::2,0]
  I[I == 144] = 0
  I[I == 109] = 0
  I[I != 0] = 1
  return I.astype(float).ravel()

def discounted_reward(rewards):
    returns = deque()
    R = 0
    for r in rewards[::-1]:
        R = r + 0.99 * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    return (returns - returns.mean()) / (returns.std())

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(80*80, 256),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

neural_net = Network()
optimizer = torch.optim.Adam(neural_net.parameters(), lr=1e-4)

env = gym.make("Pong-v0")
state = env.reset()

batch_size = 1
i_episode = 1
episode_reward = 0
running_reward = None

log_diffs, rewards = [], []

if resume:
    print("*** Load Policy Network parameters ***")
    neural_net.load_state_dict(torch.load("pong_pytorch/paramsv2_1450.pkl"))


while True:
    if render: env.render()

    x = prepro(state)
    x = torch.from_numpy(x).float().unsqueeze(0)

    prob = neural_net(x)
    y = 1 if torch.rand(1) < prob else 0
    state, reward, done, _ = env.step(y + 2)

    log_diffs.append(torch.log(abs(1 - y - prob)))   # si y=1: log(prob) et si y=0: log(1-prob)
    rewards.append(reward)

    episode_reward += reward

    if done:
        if i_episode % batch_size == 0:
            returns = discounted_reward(rewards)

            loss = 0
            for log_diff, R in zip(log_diffs, returns):
                loss += -R * log_diff

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_diffs, rewards = [], []

        running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01
        print(f"Episode {i_episode} episode_reward {episode_reward} running_reward {running_reward}")

        if i_episode % 50 == 0:
            print(f"ep {i_episode}: model saving...")
            torch.save(neural_net.state_dict(), 'params.pkl')

        state = env.reset()
        i_episode += 1
        episode_reward = 0





# fonction coût (log_loss pondérée par les rewards) : Somme(-R * log(écart entre y et prob))