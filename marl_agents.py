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
from tqdm import tqdm_notebook as tqdm

class MarelleAgent(object):
    '''
    A basic marelle agent class for both RL and non RL AIs
    '''
    def __init__(self, env):
        self.env = env
        self.n_actions = len(self.env.board.id_to_action)
        self.place_actions = 24 # 24 positions
        self.place_capture_actions = 24 # (23 captures + 1 non capture)
        self.n_total_place_actions = self.place_actions * self.place_capture_actions
        self.move_actions = 36 # 36 edges
        self.move_capture_actions = 25 # (24 captures + 1 non capture)
        self.n_total_move_actions = self.move_actions * self.move_capture_actions
        self.player_id = 0

    def act(self, state, train=False):
        return self.learned_act(state)

    def learned_act(self, state):
        raise NotImplementedError
  

class ReinforceAgent(MarelleAgent):
    ''' This class encapsulates all agents that perform reinforcement training'''
    def __init__(self, env, epsilon=0, gamma=1,  win_reward=1, defeat_reward=-1, capture_reward=0.1, captured_reward=-0.1):
        super(ReinforceAgent, self).__init__(env)
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
                a = np.random.choice(self.env.board.get_legal_action_ids(self.player_id))
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
        epoch_loop = tqdm(range(n_epoch), desc="Epochs")
        for epoch in epoch_loop:
            epoch_reward, epoch_loss = self.optimize_model(n_trajectories, opponent_agent)
            rewards.append(np.mean(epoch_reward))

            print(f'Episode {epoch + 1}/{n_epoch}: rewards {round(np.mean(epoch_reward), 2)} +/- {round(np.std(epoch_reward), 2)} - Loss : {epoch_loss}')
            
            if (epoch+1) % evaluate_freq == 0:
                    evaluation = evaluate(self.env, self, evaluation_agent, 200)
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
    def __init__(self, env, model, lr, win_reward=1, defeat_reward=-1, capture_reward=0.1, captured_reward=-0.1, epsilon=0, gamma=1):
        super(SingleModelReinforce, self).__init__(
            env=env, 
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

        self.player_id = 1
        opponent.player_id = -1
        
        trajectory_loop = tqdm(range(n_trajectories), desc="Trajectories", leave=False)
        for i in trajectory_loop:
            # Swap starting player at the middle of the training
            if i == int(n_trajectories/2):
                self.player_id = -1
                opponent.player_id = 1
            done = False
            rewards=[]

            state=self.env.reset()
            state=torch.tensor(state, dtype=torch.float)
            
            sum_lprob=0
            while not done:
                agent_reward = 0
                if self.player_id == -1:
                    # au tour de l'adversaire si l'adversaire commence
                    action=opponent.act(state, train=False)
                    state, reward, done, info = self.env.step(action)
                    agent_reward += reward["game_end"] * self.defeat_reward
                    agent_reward += reward["capture_token"] * self.captured_reward

                    if done:
                        rewards.append(agent_reward)
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
                    action = opponent.act(state, train=False)
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

    
class TripleModelReinforce(ReinforceAgent):
    '''
    An agent that uses a single model to select its action
    '''
    def __init__(self, env, model_place, model_move, model_capture, lr, win_reward=1, defeat_reward=-1, capture_reward=0.1, captured_reward=-0.1, epsilon=0, gamma=1):
        super(TripleModelReinforce, self).__init__(
            env=env, 
            epsilon=epsilon, 
            gamma=gamma, 
            win_reward=win_reward, 
            defeat_reward=defeat_reward,
            capture_reward=capture_reward, 
            captured_reward=captured_reward,
        )
        self.lr = lr
        self.model_place = model_place
        self.model_move = model_move
        self.model_capture = model_capture
        
        self.optimizer_place = torch.optim.Adam(self.model_place.parameters(), lr=lr)
        self.optimizer_move = torch.optim.Adam(self.model_move.parameters(), lr=lr)
        self.optimizer_capture = torch.optim.Adam(self.model_capture.parameters(), lr=lr)
        
        
        
        
        #TO DO FAIRE DU LEARN ACT
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
        list_sum_log_prob_place=[]
        list_sum_log_prob_move=[]
        list_sum_log_prob_capture=[]
        
        list_n_place=[]
        list_n_move=[]
        list_n_capture=[]
        self.player_id = 1
        opponent.player_id = -1
        
        trajectory_loop = tqdm(range(n_trajectories), desc="Trajectories", leave=False)
        for i in trajectory_loop:
            # Swap starting player at the middle of the training
            if i == int(n_trajectories/2):
                self.player_id = -1
                opponent.player_id = 1
            done = False
            rewards=[]

            state=self.env.reset()
            state=torch.tensor(state, dtype=torch.float)
            
           
            sum_log_prob_place=0
            sum_log_prob_move=0
            sum_log_prob_capture=0
            
            n_place=1
            n_move=1
            n_capture=1
            while not done:
                agent_reward = 0
                if self.player_id == -1:
                    #au tour de l'adversaire si l'adversaire commence
                    action=opponent.act(state)
                    state, reward, done, info = self.env.step(action)
                    agent_reward += reward["game_end"] * self.defeat_reward
                    agent_reward += reward["capture_token"] * self.captured_reward

                    if done:
                        rewards.append(agent_reward)
                        break
                    
                state=torch.tensor(state, dtype=torch.float)

                # au tour de l'agent
             
                #env.board.place_token_intermediary_state((0, 1), 1) 
                #env.board.move_token_intermediary_state(((0, 0), (0, 1), 1)
               
                def split(a,b):
                    return({k: list(zip(*g))[1] for k, g in itertools.groupby(sorted(zip(b,a)), lambda x: list(x)[0])})

                #####     
                
                legal_actions=self.env.board.get_legal_actions(self.player_id)
                tuple_1_legal_actions = [a[0] for a in legal_actions if not a[1] == None]
                tuple_2_legal_actions = [a[1] for a in legal_actions if not a[1] == None]

                #move to capture possibles quand il y a des captures possibles
                legal_intermediary_actions_to_capture=split(tuple_2_legal_actions,tuple_1_legal_actions) #dico

                legal_intermediary_actions=list(dict.fromkeys([a[0] for a in legal_actions])) 
                
                if self.env.board.phase=='place': 
                    legal_intermediary_actions_id=[l for k,l in self.env.board.place_action_to_id.items() if k in legal_intermediary_actions]
                if self.env.board.phase=='move': 
                    legal_intermediary_actions_id=[l for k,l in self.env.board.move_action_to_id.items() if k in legal_intermediary_actions]
                    
                #model_place=ConvModel_small_output(24)
                #model_move=ConvModel_small_output(32)
                
                if self.env.board.phase=='place':  
                    value_all_intermediary_actions=self.model_place(state) 
                    n_place+=1
                if self.env.board.phase=='move':  
                    value_all_intermediary_actions=self.model_move(state) 
                    n_move+=1
                    #print("lol")
                
                #print(self.env.board.phase)
                #print(value_all_intermediary_actions)
                #print(legal_intermediary_actions_id)
                value_legal_intermediary_actions = torch.index_select(value_all_intermediary_actions, 0, torch.tensor(legal_intermediary_actions_id,dtype=torch.int64))

                legal_intermediary_actions_probas = nn.Softmax(dim=0)(value_legal_intermediary_actions)
                proba_id = int(torch.multinomial(legal_intermediary_actions_probas, 1))
                
                
                if self.env.board.phase=='place':  
                    sum_log_prob_place+=legal_intermediary_actions_probas[proba_id].log()
                if self.env.board.phase=='move':  
                    sum_log_prob_move+=legal_intermediary_actions_probas[proba_id].log()
                    
     
                selected_intermediary_move=legal_intermediary_actions[proba_id] # c'est vraiment un move qui a été choisi ici pas un id
                
                # si le selected move n'est pas dans la liste alors il n'ya pas de capture.
                if selected_intermediary_move in legal_intermediary_actions_to_capture.keys():   # dans ce cas capture
                    n_capture+=1
                    legal_capture_actions = legal_intermediary_actions_to_capture[selected_intermediary_move]
                    legal_capture_actions_id = [self.env.board.capture_to_id[k] for k in legal_capture_actions]

                    if self.env.board.phase=='place':  
                        intermediate_s = self.env.board.place_token_intermediary_state(selected_intermediary_move, self.player_id)
                    if self.env.board.phase=='move':  
                        intermediate_s = self.env.board.move_token_intermediary_state(selected_intermediary_move, self.player_id)
                    
                    value_all_capture=self.model_capture(torch.tensor(intermediate_s,dtype=torch.float)) 
                    value_legal_capture = torch.index_select(value_all_capture, 0, torch.tensor(legal_capture_actions_id,dtype=torch.int64))
                    legal_capture_actions_probas = nn.Softmax(dim=0)(value_legal_capture)
                    proba_id = int(torch.multinomial(legal_capture_actions_probas, 1))
                    selected_capture=legal_capture_actions[proba_id]
                    
                    selected_action=(selected_intermediary_move,selected_capture)
                    sum_log_prob_capture+=legal_capture_actions_probas[proba_id].log()
                            
                else: 
                    selected_action=(selected_intermediary_move,None)
                
                #Définition de l'action finale
                selected_action_id = self.env.board.action_to_id[selected_action]
                state, reward, done, info =self.env.step(selected_action_id)

                agent_reward += reward["game_end"] * self.win_reward
                agent_reward += reward["capture_token"] * self.capture_reward

                # au tour de l'adversaire si agent commence
                if self.player_id == 1 and not done:
                    action = opponent.act(state)
                    state, reward, done, info = self.env.step(action)
                    agent_reward += reward["game_end"] * self.defeat_reward
                    agent_reward += reward["capture_token"] * self.captured_reward

                rewards.append(agent_reward)
                
           
            list_sum_log_prob_place.append(sum_log_prob_place)
            list_sum_log_prob_move.append(sum_log_prob_move)
            list_sum_log_prob_capture.append(sum_log_prob_capture)
            
            list_n_place.append(n_place)
            list_n_move.append(n_move)
            list_n_capture.append(n_capture)
            
            reward_trajectories.append(self._compute_returns(rewards))
        
        loss_place=0
        loss_move=0
        loss_capture=0
        
        for i in range(len(list_sum_log_prob_place)):
            loss_place+=-list_sum_log_prob_place[i]*reward_trajectories[i]*(1/list_n_place[i])
            loss_move+=-list_sum_log_prob_move[i]*reward_trajectories[i]*(1/list_n_move[i])
            loss_capture+=-list_sum_log_prob_capture[i]*reward_trajectories[i]*(1/list_n_capture[i])
        
        #PAs forcement utile de diviser ? -> Oui vu que le nolbre de trajectoire reste fixe
        loss_place=loss_place/len(list_sum_log_prob_place)
        loss_move=loss_move/len(list_sum_log_prob_place)
        loss_capture=loss_capture/len(list_sum_log_prob_place)
        #print("loss",loss)
        
        # The following lines take care of the gradient descent step for the variable loss
          
        # Discard previous gradients
        self.optimizer_place.zero_grad()
        self.optimizer_move.zero_grad()
        self.optimizer_capture.zero_grad()
        
        # Compute the gradient 
        loss_place.backward()
        loss_move.backward()
        loss_capture.backward()
       
        # Do the gradient descent step
        
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
        self.optimizer_place.step()
        self.optimizer_move.step()
        self.optimizer_capture.step()
       
       
        return reward_trajectories, loss_place


    
    
class RandomAgent(MarelleAgent):
    ''' An agent that plays randomly each turn'''
    def __init__(self, env):
        super(RandomAgent, self).__init__(env)
        pass

    def learned_act(self, s):
        return(np.random.choice(self.env.board.get_legal_action_ids(self.player_id)))


class BetterRandomAgent(MarelleAgent):
    '''An agent that captures if possible, then block if possible, then play randomly'''
    def __init__(self, env):
        super(BetterRandomAgent, self).__init__(env)
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
    