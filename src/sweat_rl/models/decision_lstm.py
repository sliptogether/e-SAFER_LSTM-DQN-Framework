import torch.nn as nn

class DecisionLSTM(nn.Module):
    def __init__(self, n_in, hidden, horizon=10):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.trajectory_head = nn.Linear(hidden, horizon)
        self.summary_head = nn.Linear(hidden, 3)

    def forward(self, x):
        z, _ = self.lstm(x)
        h = z[:, -1, :]
        return self.trajectory_head(h), self.summary_head(h)
