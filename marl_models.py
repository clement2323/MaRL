import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
import numpy as np

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
    
    
    def __init__(self,  n_actions):
        super(ConvModel, self).__init__()
        self.n_actions=n_actions
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv3 = nn.Conv2d(40, 40, kernel_size=3)
        self.fc1 = nn.Linear(40, self.n_actions)
    
    
    def transform_input_to_mat(self,state):
        
        vec_s=np.array(state.detach())     
        s=[]
        
        for u in vec_s:
            if u==0:
                s.append(-3)
            else:
                s.append(u)
        mat_state=np.ones((7,7))*(-3)
        mat_state[3,4]=s[0]
        mat_state[2,4]=s[1]
        mat_state[2,3]=s[2]
        mat_state[2,2]=s[3]

        mat_state[3,2]=s[4]
        mat_state[4,2]=s[5]
        mat_state[4,3]=s[6]
        mat_state[4,4]=s[7]

        mat_state[3,5]=s[8]
        mat_state[1,5]=s[9]
        mat_state[1,3]=s[10]
        mat_state[1,1]=s[11]

        mat_state[3,1]=s[12]
        mat_state[5,1]=s[13]
        mat_state[5,3]=s[14]
        mat_state[5,5]=s[15]

        mat_state[3,6]=s[16]
        mat_state[0,6]=s[17]
        mat_state[0,3]=s[18]
        mat_state[0,0]=s[19]

        mat_state[3,0]=s[20]
        mat_state[6,0]=s[21]
        mat_state[6,3]=s[22]
        mat_state[6,6]=s[23]
    
        return(torch.tensor(mat_state.reshape(1,1,7,7),dtype=torch.float))

    def forward(self, x):   
        
        mat_x=self.transform_input_to_mat(x)
        x=self.conv3(self.conv2(self.conv1(mat_x)))
        #print(x.shape[0]*x.shape[1]*x.shape[2])
        x = x.view(-1, 40)
        x= self.fc1(x)
        x = F.relu(x)
        return x[0]
    

    



#Poue le modele en 3 paeties à 3 réseaux
class ConvModel_small_output(nn.Module):
    
    def __init__(self,  n_actions):
        super(ConvModel_small_output, self).__init__()
        self.n_actions=n_actions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32,8, kernel_size=3)
        self.fc1 = nn.Linear(72, n_actions)
       
   
    def transform_input_to_mat(self,state):
        
        vec_s=np.array(state.detach())     
        s=vec_s
       
       # s=[]
        
        #for u in vec_s:
         #   if u==0:
          #      s.append(-3)
           # else:
            #    s.append(u)
        mat_state=np.ones((7,7))*(-3)
        mat_state[3,4]=s[0]
        mat_state[2,4]=s[1]
        mat_state[2,3]=s[2]
        mat_state[2,2]=s[3]

        mat_state[3,2]=s[4]
        mat_state[4,2]=s[5]
        mat_state[4,3]=s[6]
        mat_state[4,4]=s[7]

        mat_state[3,5]=s[8]
        mat_state[1,5]=s[9]
        mat_state[1,3]=s[10]
        mat_state[1,1]=s[11]

        mat_state[3,1]=s[12]
        mat_state[5,1]=s[13]
        mat_state[5,3]=s[14]
        mat_state[5,5]=s[15]

        mat_state[3,6]=s[16]
        mat_state[0,6]=s[17]
        mat_state[0,3]=s[18]
        mat_state[0,0]=s[19]

        mat_state[3,0]=s[20]
        mat_state[6,0]=s[21]
        mat_state[6,3]=s[22]
        mat_state[6,6]=s[23]
    
        return(torch.tensor(mat_state.reshape(1,1,7,7),dtype=torch.float))


    def forward(self, u):   
        
        x=self.transform_input_to_mat(u)        
        x = F.relu(self.conv1(x), 2)
        x = F.relu(self.conv2(x), 2)
        #print(x.shape[3]*x.shape[1]*x.shape[2])
        x = x.view(-1, 72)
        #x = F.relu(self.fc1(x))
        return self.fc1(x)[0]   
