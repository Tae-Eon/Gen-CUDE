import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_input ,d_hidden, d_output):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_hidden)
        self.fc4 = nn.Linear(d_hidden, d_hidden)
        self.fc5 = nn.Linear(d_hidden, d_hidden)
        self.fc6 = nn.Linear(d_hidden, d_hidden)
        self.fc7 = nn.Linear(d_hidden, d_output)
        
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=F.relu(self.fc6(x))
        logit=self.fc7(x)
        output=F.log_softmax(logit)
        return output