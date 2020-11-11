import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim

class FCModel(nn.Module):
    def __init__(self, dim_observation, n_actions):
        super(FCModel, self).__init__()
        
        self.n_actions = n_actions
        self.dim_observation = dim_observation
        
        self.net = nn.Sequential(
            nn.Linear(in_features=self.dim_observation, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.n_actions),
            #nn.Softmax(dim=0)
        )
        
    def forward(self, state):
        return self.net(state)
    
   # def select_action(self, state):
    #    action = torch.multinomial(self.forward(state), 1)
     #   return action
        

class ConvModel(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3)
        self.fc1 = nn.Linear(720, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 720)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
