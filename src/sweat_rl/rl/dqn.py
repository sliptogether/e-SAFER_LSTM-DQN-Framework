from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,n_action)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, size):
        self.data = deque(maxlen=size)

    def add(self, *item):
        self.data.append(item)

    def __len__(self):
        return len(self.data)

    def sample(self, n):
        b = random.sample(self.data, n)
        s,a,r,ns,d = zip(*b)
        return (
            torch.tensor(np.asarray(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.asarray(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32)
        )

def train_dqn(env, cfg):
    seed=int(cfg["project"].get("seed",42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.action_space.seed(seed)
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    q = QNetwork(n_state,n_action)
    target = QNetwork(n_state,n_action)
    target.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=float(cfg["rl_demo"]["lr"]))
    replay = ReplayBuffer(int(cfg["rl_demo"]["replay_size"]))

    gamma = float(cfg["rl_demo"]["gamma"])
    batch = int(cfg["rl_demo"]["batch_size"])
    eps = float(cfg["rl_demo"]["epsilon_start"])
    eps_end = float(cfg["rl_demo"]["epsilon_end"])
    eps_decay = float(cfg["rl_demo"]["epsilon_decay"])
    target_update = int(cfg["rl_demo"]["target_update"])

    rewards = []
    for ep in range(int(cfg["rl_demo"]["episodes"])):
        s,_ = env.reset()
        done = False
        total = 0.0

        while not done:
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = int(torch.argmax(q(torch.tensor(s[None],dtype=torch.float32)),1).item())

            ns,r,terminated,truncated,_ = env.step(a)
            done = terminated or truncated
            replay.add(s,a,r,ns,done)
            s = ns
            total += r

            if len(replay) >= batch:
                bs,ba,br,bns,bd = replay.sample(batch)
                qp = q(bs).gather(1,ba[:,None]).squeeze(1)
                with torch.no_grad():
                    y = br + gamma*(1-bd)*target(bns).max(1).values
                loss = torch.mean((qp-y)**2)
                opt.zero_grad()
                loss.backward()
                opt.step()

        rewards.append(total)
        eps = max(eps_end, eps*eps_decay)
        if (ep+1)%target_update == 0:
            target.load_state_dict(q.state_dict())

    return q, rewards


def infer_policy(q, normalized_states):
    states=torch.tensor(np.asarray(normalized_states),dtype=torch.float32)
    q.eval()
    with torch.no_grad():
        q_values=q(states)
        scores=torch.softmax(q_values,dim=1)
        actions=torch.argmax(q_values,dim=1)
    return actions.numpy(),q_values.numpy(),scores.numpy()
