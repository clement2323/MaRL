from marl_evaluations import evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import wandb
import os

class MarelleAgent(object):
    '''
    A basic marelle agent class for both RL and non RL AIs
    '''
    def __init__(self, env, player_id):
        self.env = env
        self.n_actions = len(self.env.board.id_to_action)
        self.place_actions = 24 # 24 positions
        self.place_capture_actions = 24 # (23 captures + 1 non capture)
        self.n_total_place_actions = self.place_actions * self.place_capture_actions
        self.move_actions = 36 # 36 edges
        self.move_capture_actions = 25 # (24 captures + 1 non capture)
        self.n_total_move_actions = self.move_actions * self.move_capture_actions
        self.player_id = player_id

    def act(self, state):
        return self.learned_act(state)

    def learned_act(self, state):
        raise NotImplementedError
  

class ReinforceAgent(MarelleAgent):
    ''' This class encapsulates all agents that perform reinforcement training'''
    def __init__(self, env, player_id, epsilon=0, gamma=1,  win_reward=1, defeat_reward=-1, capture_reward=0.1, captured_reward=-0.1):
        super(ReinforceAgent, self).__init__(env, player_id)
        self.epsilon = epsilon
        self.gamma = gamma
        self.win_reward = win_reward
        self.defeat_reward = defeat_reward
        self.capture_reward = capture_reward
        self.captured_reward = captured_reward
        self.model = None

    def act(self,s,train=True):
        """ This function should return the next action to do:
        an integer between 0 and 4 (not included) with a random exploration of epsilon"""
        if train:
            if np.random.rand() <= self.epsilon:
                a = np.random.randint(0, self.n_actions, size=1)[0]
            else:
                a = self.learned_act(s)
        else: # in some cases, this can improve the performance.. remove it if poor performances
            a = self.learned_act(s)

        return a
    
    def train(self, n_trajectories, n_epoch, opponent_agent, evaluation_agent, log_wandb=False, save_model_freq = 50, evaluate_freq = 25):
        """Training method

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expected gradient
        n_update : int
            The number of gradient updates
            
        """
        rewards = []
        for epoch in range(n_epoch):
            epoch_reward, epoch_loss = self.optimize_model(n_trajectories, opponent_agent)
            rewards.append(np.mean(epoch_reward))

            print(f'Episode {epoch + 1}/{n_epoch}: rewards {round(np.mean(epoch_reward), 2)} +/- {round(np.std(epoch_reward), 2)} - Loss : {epoch_loss}')
            
            if (epoch+1) % evaluate_freq == 0:
                    evaluation = evaluate(self.env, self, evaluation_agent, 100, self.player_id)
                    print(evaluation)
            
            if log_wandb:
                wandb_log = {"episode": epoch + 1, "rewards" : round(np.mean(epoch_reward), 2), "+/-": round(np.std(epoch_reward), 2), "loss": epoch_loss}
                
                if (epoch+1) % evaluate_freq == 0:
                    for key in evaluation:
                        wandb_log[key] = evaluation[key]
                
                wandb.log(wandb_log)
                if (epoch+1) % save_model_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, f'model_{epoch + 1}_{n_epoch}.pt'))    

        if log_wandb:    
            torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, 'model_final.pt'))

        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], [rewards[i]]) for i in range(len(rewards))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd')
   

    def learned_act(self,s):
        """ Act via the policy of the agent, from a given state s
        it proposes an action a"""
        raise NotImplementedError

    
    def _compute_returns(self, rewards):

        returns=0
        for i in range(len(rewards)):
            returns=self.gamma*returns+rewards[-(i+1)]
        return(returns)
        
    def optimize_model(self, n_trajectories, adversaire):
        """Perform a gradient update using n_trajectories

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expectation card(D) in the formula above

        Returns
        -------
        array
            The cumulative discounted rewards of each trajectory
        """
        
        raise NotImplementedError 
    
    def save(self):
        """ This function returns basic stats if applicable: the
        loss and/or the model"""
        pass

    def load(self):
        """ This function allows to restore a model"""
        pass


class SingleModelReinforce(ReinforceAgent):
    '''
    An agent that uses a single model to select its action
    '''
    def __init__(self, env, player_id, model, lr, win_reward=1, defeat_reward=-1, capture_reward=0.1, captured_reward=-0.1, epsilon=0, gamma=1):
        super(SingleModelReinforce, self).__init__(
            env=env, 
            player_id=player_id, 
            epsilon=epsilon, 
            gamma=gamma, 
            win_reward=win_reward, 
            defeat_reward=defeat_reward,
            capture_reward=capture_reward, 
            captured_reward=captured_reward,
        )
        self.lr = lr
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def learned_act(self, s): #checker legal move + argmax
        s=torch.tensor(s,dtype=torch.float)
        legal_moves = self.env.board.get_legal_action_ids(self.player_id)
        t_all_moves=np.array(self.model(s).detach())
        t_legal_moves =[t_all_moves[legal_move] for legal_move in legal_moves] #pas besoin de softmaxiser ici
        argm = np.argmax(t_legal_moves)
        action = legal_moves[argm]

        return(action) 
               
    def optimize_model(self, n_trajectories, opponent:MarelleAgent):
      
        reward_trajectories=[]
        list_sum_proba=[]
        
        #Here I compute n_trajectories trajectories in order to calculate the MonteCarlo estimate of the J function
        for i in range(n_trajectories):
            done = False
            rewards=[]

            state=self.env.reset()
            state=torch.tensor(state, dtype=torch.float)
            
            sum_lprob=0
            while not done:
                agent_reward = 0
                if self.player_id == 2:
                    #au tour de l'adversaire si l'adversaire commence
                    action=opponent.act(state)
                    state, reward, done, info = self.env.step(action)
                    agent_reward += reward["game_end"] * self.defeat_reward
                    agent_reward += reward["capture_token"] * self.captured_reward

                    if done:
                        break
                    
                state=torch.tensor(state, dtype=torch.float)

                # au tour de l'agent
                legal_moves = self.env.board.get_legal_action_ids(self.player_id)
                t_legal_moves = torch.tensor(legal_moves, dtype=torch.int64)
                t_all_moves = self.model(state)

                t_legal_moves_scores = torch.index_select(t_all_moves, 0, t_legal_moves)
                
                # Softmax on legal moves
                t_legal_moves_probas = nn.Softmax(dim=0)(t_legal_moves_scores)

                proba_id = int(torch.multinomial(t_legal_moves_probas, 1))
                action_id = legal_moves[proba_id]
                
                # La proba est la proba du softmax des legal moves
                lprob = t_legal_moves_probas[proba_id].log()
                sum_lprob+= lprob
                
                state, reward, done, info =self.env.step(action_id)

                agent_reward += reward["game_end"] * self.win_reward
                agent_reward += reward["capture_token"] * self.capture_reward


                # au tour de l'adversaire si agent commence
                if self.player_id == 1 and not done:
                    action = opponent.act(state)
                    state, reward, done, info = self.env.step(action)
                    agent_reward += reward["game_end"] * self.defeat_reward
                    agent_reward += reward["capture_token"] * self.captured_reward

                rewards.append(agent_reward)
            #print("sumlprob",sum_lprob)
            list_sum_proba.append(sum_lprob)
            reward_trajectories.append(self._compute_returns(rewards))
        
        loss=0
        
        #print("vecteur sum prob pour chaque trajectoire")
        #t_s_p=np.array([l.detach() for l in list_sum_proba])
        #print("max",np.max(t_s_p))
        #print("min",np.min(t_s_p))
        
        for i in range(len(list_sum_proba)):
            loss+=-list_sum_proba[i]*reward_trajectories[i]
        
        loss=loss/len(list_sum_proba)
        #print("loss",loss)
        
        # The following lines take care of the gradient descent step for the variable loss
          
        # Discard previous gradients
        self.optimizer.zero_grad()
        
        # Compute the gradient 
        loss.backward()
       
        # Do the gradient descent step
        casse=False
        for index, weight in enumerate(self.model.parameters()):
            gradient, *_ = weight.grad.data
            gradient=torch.isfinite(gradient)
            #print(gradient)
            gradient=np.array(gradient)
            if np.any(gradient) == False :
                casse=True
            #print(f"Gradient of w{index} w.r.t to L: {gradient}")
              
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
        if(casse):
            print("explosion, go à l'époque suivante")
        else:
            self.optimizer.step()
       
        return reward_trajectories, loss



class RandomAgent(MarelleAgent):
    ''' An agent that plays randomly each turn'''
    def __init__(self, env, player_id):
        super(RandomAgent, self).__init__(env, player_id)
        pass

    def learned_act(self, s):
        return(np.random.choice(self.env.board.get_legal_action_ids(self.player_id)))


class BetterRandomAgent(MarelleAgent):
    '''An agent that captures if possible, then block if possible, then play randomly'''
    def __init__(self, env, player_id):
        super(BetterRandomAgent, self).__init__(env, player_id)
        pass

    def learned_act(self, s):
        legal_actions = self.env.board.get_legal_actions(self.player_id)
        opponent_legal_actions = self.env.board.get_legal_actions(self.env.board.get_opponent(self.player_id))
        
        pos_to_block = []
        for opponent_action in opponent_legal_actions:
            # Check positions we should block
            pos, capture_pos = opponent_action
            if capture_pos != None:
                pos_to_block.append(pos)
        blocker_action = None
        for action in legal_actions:
            # Check if possible to capture
            pos, enemy_pos = action
            if enemy_pos != None:
                return self.env.board.action_to_id[action]
            
            # Check if possible to block
            if self.env.board.phase == "place":
                if pos in pos_to_block:
                    blocker_action = action
            
            elif self.env.board.phase == "move":
                a, b = pos
                if a in pos_to_block or b in pos_to_block:
                    blocker_action = action
        
        if blocker_action != None:
            return self.env.board.action_to_id[blocker_action]
        
        return(np.random.choice(self.env.board.get_legal_action_ids(self.player_id)))
    