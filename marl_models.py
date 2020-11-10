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