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

class Agent(object):
    def __init__(self, env, player_id, epsilon=0):
        self.epsilon = epsilon
        self.env = env
        self.n_action = len(self.env.board.id_to_action)
        self.gamma = 1
        self.player_id = player_id
        
    def set_epsilon(self,e):
        self.epsilon = e

    def act(self,s,train=True):
        """ This function should return the next action to do:
        an integer between 0 and 4 (not included) with a random exploration of epsilon"""
        if train:
            if np.random.rand() <= self.epsilon:
                a = np.random.randint(0, self.n_action, size=1)[0]
            else:
                a = self.learned_act(s)
        else: # in some cases, this can improve the performance.. remove it if poor performances
            a = self.learned_act(s)

        return a

    def learned_act(self,s):
        """ Act via the policy of the agent, from a given state s
        it proposes an action a"""
        pass

    
    def _compute_returns(self, rewards):
        """Returns the cumulative discounted rewards at each time step
        
        Parameters
        ----------
        rewards : array
            The array of rewards of one episode

        Returns
        -------
        array
            The cumulative discounted rewards at each time step
            
        Example
        -------
        for rewards=[1, 2, 3] this method outputs [1 + 2 * gamma + 3 * gamma**2, 2 + 3 * gamma, 3] 
        """
        raise NotImplementedError
        
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
            rewards.append(epoch_reward)
            if (epoch+1) % 5 == 0:
                print(f'Episode {epoch + 1}/{n_epoch}: rewards {round(np.mean(rewards[-1]), 2)} +/- {round(np.std(rewards[-1]), 2)} - Loss : {epoch_loss}')
            
            if (epoch+1) % evaluate_freq == 0:
                    evaluation = evaluate(self.env, self, evaluation_agent, 100, self.player_id)
                    # print(dict(enumerate(evaluation)))
                    print(evaluation)
            
            if log_wandb:
                wandb_log = {"episode": epoch + 1, "rewards" : round(np.mean(rewards[-1]), 2), "+/-": round(np.std(rewards[-1]), 2), "loss": epoch_loss}
                
                if (epoch+1) % evaluate_freq == 0:
                    for key in evaluation:
                        wandb_log[key] = evaluation[key]
                
                wandb.log(wandb_log)
                if (epoch+1) % save_model_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, f'model_{epoch + 1}_{n_epoch}.pt'))

                
                    

        if log_wandb:    
            torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, 'model_final.pt'))

        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards[i]) for i in range(len(rewards))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd')
    
    
    def save(self):
        """ This function returns basic stats if applicable: the
        loss and/or the model"""
        pass

    def load(self):
        """ This function allows to restore a model"""
        pass


class Reinforce_cliff(): #servira pour les environnements  en tour par tour
    
    def __init__(self, env, model, lr):
        self.env = env
        self.gamma = 1
        self.lr = lr
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=lr)
        self.n_action = self.env.Na

        
    def learned_act(self, prev_state, state): #checker legal move + argmax
        
        proba=nn.Softmax(dim=0)(self.model(torch.tensor([prev_state,state],dtype = torch.float)))
        proba=np.array(proba.detach())
        action = np.argmax(proba)

        return(action) 
        
    def _compute_returns(self, rewards):
        
        returns=0
        for i in range(len(rewards)):
            returns=self.gamma*returns+rewards[-(i+1)]
        return(returns)
               
    def optimize_model(self, n_trajectories): #adversaire on met dedans l'agent qui jouera contre
      
        reward_trajectories=[]
        list_sum_proba=[]
        
        #Here I compute n_trajectories trajectories in order to calculate the MonteCarlo estimate of the J function
        for i in range(n_trajectories):
            done = False
            rewards=[]
            
            state=self.env.reset()
            state=torch.tensor(state, dtype=torch.float)
            prev_state = state
            sum_lprob=0
            
            while not done:
                
                proba=nn.Softmax(dim=0)(self.model(torch.tensor([prev_state,state],dtype = torch.float)))
                
                action=int(torch.multinomial(proba, 1))
                sum_lprob+= proba[action].log()     
                prev_state = state
                state, reward, done, info =self.env.step(action)
                rewards.append(reward)
                state=torch.tensor(state, dtype=torch.float)
            
            list_sum_proba.append(sum_lprob)
            reward_trajectories.append(self._compute_returns(rewards))
        
        loss=0
        
      
        for i in range(len(list_sum_proba)):
            loss+=-list_sum_proba[i]*reward_trajectories[i]
        
        loss=loss/len(list_sum_proba)
      
        # Discard previous gradients
        self.optimizer.zero_grad()
        
        # Compute the gradient 
        loss.backward()
       
        # Do the gradient descent step
        self.optimizer.step()
        
        return reward_trajectories, loss

    def train(self, n_trajectories, n_epoch, adversaire, log_wandb=False, save_model_freq = 50):
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
            epoch_reward, epoch_loss = self.optimize_model(n_trajectories)
            rewards.append(epoch_reward)
            if (epoch+1)%save_model_freq == 0:
                print(f'Episode {epoch + 1}/{n_epoch}: rewards {round(np.mean(rewards[-1]), 2)} +/- {round(np.std(rewards[-1]), 2)} - Loss : {epoch_loss}')
            
            if log_wandb:
                wandb.log({"episode": epoch + 1, "rewards" : round(np.mean(rewards[-1]), 2), "+/-": round(np.std(rewards[-1]), 2), "loss": epoch_loss})
                if (epoch+1) % save_model_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, f'model_{epoch + 1}_{n_epoch}.pt'))

        if log_wandb:    
            torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, 'model_final.pt'))

        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards[i]) for i in range(len(rewards))))), columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd')
    




    

    
    
class Reinforce_place(Agent):## refaire ça en changeant l'environnement ?
    
    def __init__(self, env, player_id, model, lr, incentivize_captures=False, punish_opponent_captures=False):
        super(Reinforce_place, self).__init__(env, player_id, model)
        self.lr = lr
        self.model = model
        self.incentivize_captures = incentivize_captures
        self.punish_opponent_captures = punish_opponent_captures
        self.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=lr)
        self.n_action = 566
        
    def learned_act(self, s): #checker legal move + argmax
        s = torch.tensor(s, dtype=torch.float32)
        legal_moves = self.env.board.get_legal_action_ids(self.player_id)
        #print(legal_moves)
        s=torch.tensor(s,dtype=torch.float)
        legal_moves = np.array(legal_moves)[np.array(legal_moves)<=565] 
        t_all_moves=np.array(self.model(s).detach())   
        #t_all_moves=np.array(self.model(s).detach())
        t_legal_moves =[t_all_moves[legal_move] for legal_move in legal_moves] #pas besoin de softmaxiser ici
        argm = np.argmax(t_legal_moves)
        action = legal_moves[argm]

        return(action) 
        
    def _compute_returns(self, rewards):
        
        returns=0
        for i in range(len(rewards)):
            returns=self.gamma*returns+rewards[-(i+1)]
        return(returns)
               
    def optimize_model(self, n_trajectories, adversaire): #adversaire on met dedans l'agent qui jouera contre
      
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
                
                  
                legal_moves = self.env.board.get_legal_action_ids(self.player_id)
                #print(legal_moves)
                legal_moves = np.array(legal_moves)[np.array(legal_moves)<=565]
                
                t_all_moves=self.model(state)
                t_legal_moves =torch.tensor([t_all_moves[legal_move] for legal_move in legal_moves])
                
                #softmax ici
                t_legal_moves = nn.Softmax(dim=0)(t_legal_moves)
                action_id = int(torch.multinomial(t_legal_moves, 1))
                action = legal_moves[action_id]
                
                #je choisi un legal move mais c'est la proba de l'action que je prends en espérant qu'elle ne vaille pas 0
                proba=nn.Softmax(dim=0)(self.model(state))
                sum_lprob+= proba[action].log()             
                state, reward, done, info =self.env.step(action)
                
                #done  = True if end of place
                done = (self.env.board.phase=='move')
                
                reward_p1 = reward["end_place_phase"]
                if self.incentivize_captures:
                    reward_p1 += reward["capture_token"] * 0.1
                
                #au tour de l'adversaire
                if not done:
                    action=adversaire.act(state)
                    state, reward, done, info = self.env.step(action)
                    
                    #done  = True if end of place
                    done = (self.env.board.phase=='move')
                    reward_p2 = reward["end_place_phase"]
                    if self.punish_opponent_captures:
                        reward_p2 += reward["capture_token"] * 0.1
                    rewards.append(reward_p1-reward_p2)
                else : 
                    rewards.append(reward_p1)
                    
                state=torch.tensor(state, dtype=torch.float)
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



class Reinforce(Agent):
    def __init__(self, env, player_id, model, lr, incentivize_captures=False, punish_opponent_captures=False, domain = None ):
        super(Reinforce, self).__init__(env, player_id, model)
        self.lr = lr
        self.model = model
        self.incentivize_captures = incentivize_captures
        self.punish_opponent_captures = punish_opponent_captures
        self.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=lr)
        self.domain = domain
        
    def learned_act(self, s): #checker legal move + argmax
        s=torch.tensor(s,dtype=torch.float)
        legal_moves = self.env.board.get_legal_action_ids(self.player_id)
        t_all_moves=np.array(self.model(s).detach())
        t_legal_moves =[t_all_moves[legal_move] for legal_move in legal_moves] #pas besoin de softmaxiser ici
        argm = np.argmax(t_legal_moves)
        action = legal_moves[argm]

        return(action) 
        
    def _compute_returns(self, rewards):
        
        returns=0
        for i in range(len(rewards)):
            returns=self.gamma*returns+rewards[-(i+1)]
        return(returns)
               
    def optimize_model(self, n_trajectories, adversaire): #adversaire on met dedans l'agent qui jouera contre
      
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
                
                  
                legal_moves = self.env.board.get_legal_action_ids(self.player_id)
                t_all_moves=self.model(state)
                t_legal_moves =torch.tensor([t_all_moves[legal_move] for legal_move in legal_moves])
                
                #softmax ici
                t_legal_moves = nn.Softmax(dim=0)(t_legal_moves)
                #print("num traj",i)
                #print("t_legal_moves",t_legal_moves)
                action_id = int(torch.multinomial(t_legal_moves, 1))
                action = legal_moves[action_id]
                
                #je choisi un legal move mais c'est la proba de l'action que je prends en espérant qu'elle ne vaille pas 0
                proba=nn.Softmax(dim=0)(self.model(state))
                sum_lprob+= proba[action].log()
                #print("proba",proba[action].log())
                
                state, reward, done, info =self.env.step(action)

                reward_p1 = reward["game_end"]
                if self.incentivize_captures:
                    reward_p1 += reward["capture_token"] * 0.1
                       
                #au tour de l'adversaire
                if not done:
                    action=adversaire.act(state)
                    state, reward, done, info = self.env.step(action)
                    reward_p2 = reward["game_end"]
                    if self.punish_opponent_captures:
                        reward_p2 += reward["capture_token"] * 0.1
                    rewards.append(reward_p1-reward_p2)
                else : 
                    rewards.append(reward_p1)
                    
                state=torch.tensor(state, dtype=torch.float)
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



class RandomAgent(Agent):
    def __init__(self, env, player_id):
        super(RandomAgent, self).__init__(env, player_id)
        pass

    def learned_act(self, s):
        return(np.random.choice(self.env.board.get_legal_action_ids(self.player_id)))


class BetterRandomAgent(Agent):
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
    