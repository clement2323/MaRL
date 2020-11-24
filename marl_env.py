import networkx as nx
import gym
from gym import spaces
from IPython.display import clear_output
from termcolor import colored
from tqdm import tqdm_notebook as tqdm
import numpy as np

class MarelleGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, end_after_place_phase=False):
        super(MarelleGymEnv, self).__init__()    
        self.board = MarelleBoard()
        self.end_after_place_phase = end_after_place_phase
        self.current_player = 1
        self.N_STATE = 24 # 24 positions
        self.N_PLACE_ACTIONS = 24 # 24 positions
        self.N_PLACE_CAPTURE_ACTIONS = 24 # 23 captures + 1 non capture
        self.N_MOVE_ACTIONS = 32 # 36 edges
        self.N_MOVE_CAPTURE_ACTIONS = 25 # 24 captures + 1 non capture
        self.N_TOTAL_PLACE_ACTIONS = self.N_PLACE_ACTIONS * self.N_PLACE_CAPTURE_ACTIONS
        self.N_TOTAL_MOVE_CAPTURE_ACTIONS = self.N_MOVE_ACTIONS * self.N_MOVE_CAPTURE_ACTIONS
        self.N_TOTAL_ACTIONS = self.N_TOTAL_PLACE_ACTIONS + self.N_TOTAL_MOVE_CAPTURE_ACTIONS
        
    def step(self, action_id): 
   
        self.board.play_action(action_id)
        
        reward = {}
        end_check_status = self.board.check_if_end()

        observation=self.board.get_state()
        if self.end_after_place_phase:
            done = self.board.phase != "place"
        else:
            done = (end_check_status != 0)
        
        if end_check_status == 99: # draw
            reward["game_end"] = 0
        else:
            reward["game_end"] = end_check_status * self.current_player
        
        _, opponent_pos = self.board.id_to_action[action_id]
        if opponent_pos != None:
            reward["capture_token"] = 1
        else:
            reward["capture_token"] = 0
            
        info = ""
        self.current_player = self.board.get_opponent(self.current_player)

        return observation, reward, done, info
    
    # Execute one time step within the environment
  
    def reset(self):
        self.current_player = 1

        self.board.initialize_game()

        return self.board.get_state()
    # Reset

    def render(self, action_highlight=None, mode='human', close=False):
        self.board.print_board(action_highlight)


class MarelleBoard():
    '''
    TODO : Add rule to prevent removing a token in a line when capturing
    '''
    def __init__(self):
        self.N_TOKENS_PER_PLAYER = 9
        self.maximum_actions_before_draw = 300
        graph = nx.Graph()
        for i in range(3):
            for j in range(8):
                graph.add_node((i, j))
                graph.nodes[(i, j)]["state"] = 0

        for i in range(3):
            for j in range(8):
                nj = j + 1
                if nj >= 8:
                    nj = 0
                graph.add_edge((i, j), (i, nj))
                if i != 2 and (j % 2) == 0:
                    graph.add_edge((i, j), (i+1, j))
        
        self.graph = graph

        # List of all placement actions
        place_token_action_list = []
        for node_i in self.graph.nodes:
            place_token_action_list.append((node_i, None))
            for node_j in self.graph.nodes:
                if node_i != node_j:
                    place_token_action_list.append((node_i, node_j))

        self.place_token_action_list = place_token_action_list

        # List of all token movement actions
        move_token_action_list = []
        for edge in self.graph.edges:
            move_token_action_list.append((edge, None))
            for node in self.graph.nodes:
                move_token_action_list.append((edge, node))
        
        self.move_token_action_list = move_token_action_list

        id_to_action = {}
        action_to_id = {}
        i = 0
        for action in self.place_token_action_list:
            id_to_action[i] = action
            action_to_id[action] = i
            i += 1

        for action in self.move_token_action_list:
            id_to_action[i] = action
            action_to_id[action] = i
            i += 1
        
        self.id_to_action = id_to_action
        self.action_to_id = action_to_id
        
        #AJOUT POUR TRIPLE AGENT
 
        #creation d'un dictionnaire de intermediary_place to place_id
        place_action_list=self.place_token_action_list
        tuple_1 = [a[0] for a in place_action_list] #NONE ? 

        id_to_place_action = {i:j  for i, j in enumerate(list(dict.fromkeys(tuple_1)))}
        place_action_to_id = {j:i  for i, j in id_to_place_action.items()}
        
        self.id_to_place_action = id_to_place_action
        self.place_action_to_id = place_action_to_id
        #creation d'un dictionnaire de intermediary_move to move_id
        move_action_list=self.move_token_action_list
        tuple_1 = [a[0] for a in move_action_list] #NONE ? 

        id_to_move_action = {i:j  for i, j in enumerate(list(dict.fromkeys(tuple_1)))}
        move_action_to_id = {j:i  for i, j in id_to_move_action.items()}
        
        self.id_to_move_action = id_to_move_action
        self.move_action_to_id = move_action_to_id
        #captures to id
        compteur=0
        capture_to_id = {} 
        id_to_capture = {}

        for i in range(3):
            for j in range(8):
                id_to_capture[compteur]= (i,j)
                compteur+=1
        capture_to_id = {j:i for i,j in id_to_capture.items()}

        self.id_to_capture = id_to_capture
        self.capture_to_id = capture_to_id

        self.initialize_game()


    def initialize_game(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]["state"] = 0

        self.phase = "place"
        self.players = {1: {}, -1: {}}
        self.players[1]["played_tokens"] = 0
        self.players[-1]["played_tokens"] = 0
        self.players[1]["tokens_on_board"] = 0
        self.players[-1]["tokens_on_board"] = 0
        self.current_action = 0
        self.current_player = 1
        self.update_legal_actions()


    def color_value(self, player, position, highlight_positions):
        if player == 1:
            color = "red"
        elif player == 2:
            color = "blue"
        else:
            color = "white"
        
        if position in highlight_positions:
            return colored(player, color=color, on_color="on_yellow")
        
        else:
            return colored(player, color=color)


    def print_board(self, action_highlight=None):        
        action_positions = []

        if action_highlight != None:
            action_position, action_capture = action_highlight

            a, b = action_position
            # if move type = action
            if type(a) is tuple:
                action_positions.append(a)
                action_positions.append(b)
            else:
                action_positions.append((a, b))
            
            action_positions.append(action_capture)

        v = {}
        for node in self.graph.nodes:
            if self.graph.nodes[node]["state"] == 1:
                v[node] = self.color_value(1, node, action_positions)
            elif self.graph.nodes[node]["state"] == -1:
                v[node] = self.color_value(2, node, action_positions)
            else:
                v[node] = self.color_value(0, node, action_positions)


        board_grid = f"""
        Phase : {self.phase}
        Placed tokens : P1 : {self.players[1]["played_tokens"]} / P2 : {self.players[-1]["played_tokens"]} 
        Tokens on board : P1 : {self.players[1]["tokens_on_board"]} / P2 : {self.players[-1]["tokens_on_board"]} 
        {v[(2,3)]}--------------{v[(2,2)]}--------------{v[(2,1)]}
        |              |              |
        |    {v[(1,3)]}---------{v[(1,2)]}---------{v[(1,1)]}    |
        |    |         |         |    |
        |    |    {v[(0,3)]}----{v[(0,2)]}----{v[(0,1)]}    |    |
        |    |    |         |    |    |
        {v[(2,4)]}----{v[(1,4)]}----{v[(0,4)]}         {v[(0,0)]}----{v[(1,0)]}----{v[(2,0)]}                 
        |    |    |         |    |    |
        |    |    {v[(0,5)]}----{v[(0,6)]}----{v[(0,7)]}    |    |
        |    |         |         |    |
        |    {v[(1,5)]}---------{v[(1,6)]}---------{v[(1,7)]}    |
        |              |              |
        {v[(2,5)]}--------------{v[(2,6)]}--------------{v[(2,7)]}
        """
        print(board_grid)

    def play_action(self, action_id):
        if self.phase == "place":
            self.place_token_action(self.id_to_action[action_id], self.current_player)
            
        else:
            self.move_token_action(self.id_to_action[action_id], self.current_player)
        
        self.current_action += 1
        self.current_player *= -1
        self.change_phase_if_needed()
        self.update_legal_actions()
        # print(f"{self.phase} - player {self.current_player} - legal actions : {self.players[self.current_player]['legal_actions']}")
        self.check_if_end()
        # print(self.check_if_end())
    
    def update_legal_actions(self):
        for player in [1, -1]:
            legal_actions = []

            if self.phase == "place":
                legal_actions = self.place_token_legal_actions(player)
            
            elif self.phase == "move":
                legal_actions = self.move_token_legal_actions(player)

            legal_action_ids = []
            for legal_action in legal_actions:
                legal_action_ids.append(self.action_to_id[legal_action])

            self.players[player]["legal_actions"] = legal_actions
            self.players[player]["legal_action_ids"] = legal_action_ids
    
    def get_legal_actions(self, player):
        return self.players[player]["legal_actions"]
    
    def get_legal_action_ids(self, player):
        return self.players[player]["legal_action_ids"]

    def get_state(self):
        state = []
        for node in self.graph.nodes:
            state.append(self.graph.nodes[node]["state"])

        return state
    
    def place_token_action(self, action, player):

        legal_moves = self.get_legal_actions(player)

        if action not in legal_moves:
            raise Exception('Illegal place move')

        (new_token_pos, opponent_token_pos) = action
        self.graph.nodes[new_token_pos]["state"] = player
        self.players[player]["played_tokens"] += 1
        self.players[player]["tokens_on_board"] += 1


        if opponent_token_pos != None:
            self.graph.nodes[opponent_token_pos]["state"] = 0
            self.players[self.get_opponent(player)]["tokens_on_board"] -= 1
    
    def place_token_intermediary_state(self, place_position, player):
        initial_state = self.graph.nodes[place_position]["state"]
        self.graph.nodes[place_position]["state"] = player
        intermediary_state = self.get_state()
        self.graph.nodes[place_position]["state"] = initial_state
        return intermediary_state

    def move_token_intermediary_state(self, move_edge, player):
        a, b = move_edge
        state_a = self.graph.nodes[a]["state"]
        state_b = self.graph.nodes[b]["state"]
        if state_a == player:
            self.graph.nodes[a]["state"] = 0
            self.graph.nodes[b]["state"] = player
            intermediary_state = self.get_state()
            self.graph.nodes[a]["state"] = player
            self.graph.nodes[b]["state"] = state_b
        elif state_b == player:
            self.graph.nodes[b]["state"] = 0
            self.graph.nodes[a]["state"] = player
            intermediary_state = self.get_state()
            self.graph.nodes[b]["state"] = player
            self.graph.nodes[a]["state"] = state_a
        else:
            intermediary_state = self.get_state()
        
        return intermediary_state

    def place_token_legal_actions(self, player):
        opponent_nodes = []
        for node in self.graph.nodes:
            if self.graph.nodes[node]["state"] == self.get_opponent(player):
                opponent_nodes.append(node)

        legal_actions = []
        for node in self.graph.nodes:
            if self.graph.nodes[node]["state"] == 0:
                self.graph.nodes[node]["state"] = player
                
                n_capturable_opponent_nodes = 0
                if self.check_if_capture(node, player):
                    for opponent_node in opponent_nodes:
                        if not self.check_if_capture(opponent_node, self.get_opponent(player)): # prevent removing lined tokens
                            legal_actions.append((node, opponent_node))
                            n_capturable_opponent_nodes += 1

                if n_capturable_opponent_nodes == 0:
                    legal_actions.append((node, None))
                self.graph.nodes[node]["state"] = 0
        
        return legal_actions
    
    def move_token_action(self, action, player):
        legal_moves = self.get_legal_actions(player)
        if action not in legal_moves:
            raise Exception('Illegal token move action')

        edge, opponent_token = action
        a, b = edge
        if self.graph.nodes[a]["state"] == 0 and self.graph.nodes[b]["state"] == player:
            cur_node = b
            new_node = a
        elif self.graph.nodes[a]["state"] == player and self.graph.nodes[b]["state"] == 0:
            cur_node = a
            new_node = b
        else:
            raise Exception('Illegal token move action')
        
        self.graph.nodes[cur_node]["state"] = 0
        self.graph.nodes[new_node]["state"] = player

        if opponent_token != None:
            self.graph.nodes[opponent_token]["state"] = 0
            self.players[self.get_opponent(player)]["tokens_on_board"] -= 1
    
    def move_token_legal_actions(self, player):
        opponent_nodes = []
        for node in self.graph.nodes:
            if self.graph.nodes[node]["state"] == self.get_opponent(player):
                opponent_nodes.append(node)
        
        legal_actions = []

        for edge in self.graph.edges:
            a, b = edge
            if self.graph.nodes[a]["state"] == 0 and self.graph.nodes[b]["state"] == player:
                cur_node = b
                new_node = a
            elif self.graph.nodes[a]["state"] == player and self.graph.nodes[b]["state"] == 0:
                cur_node = a
                new_node = b
            else:
                continue
            self.graph.nodes[new_node]["state"] = player
            self.graph.nodes[cur_node]["state"] = 0
            
            n_capturable_opponent_nodes = 0
            if self.check_if_capture(new_node, player):
                for opponent_node in opponent_nodes:
                    if not self.check_if_capture(opponent_node, self.get_opponent(player)): # prevent removing lined tokens
                        legal_actions.append((edge, opponent_node))
                        n_capturable_opponent_nodes += 1
            if n_capturable_opponent_nodes == 0:
                legal_actions.append((edge, None))
            self.graph.nodes[new_node]["state"] = 0
            self.graph.nodes[cur_node]["state"] = player
            
        return legal_actions
    
    def check_if_capture(self, position, player):
        r, t = position
        capture = False

        # step 1, horizontal (same t, different r)
        if (t % 2 == 0):
            r_capture = True
            r_lines = [0, 1, 2]
            for line_r in r_lines:
                if self.graph.nodes[line_r, t]["state"] != player:
                    r_capture = False
                    break
            if r_capture == True:
                return True

        # step 2 : vertical (same r, different t)
        t_capture = False
        t_lines = [[7, 0, 1], [1, 2, 3], [3, 4, 5], [5, 6, 7]]
        for t_line in t_lines:
            if t not in t_line:
                continue
            
            t_capture = True
            for line_t in t_line:
                if self.graph.nodes[r, line_t]["state"] != player:
                    t_capture = False
                    break
            
            if t_capture == True:
                return True

        return False
    
    def change_phase_if_needed(self):
        if self.players[1]["played_tokens"] == self.N_TOKENS_PER_PLAYER and self.players[-1]["played_tokens"] == self.N_TOKENS_PER_PLAYER:
            self.phase = "move"

    def check_if_end(self):
        '''
        Returns 0 if the game is not ended and the id of the player if a player won
        '''
        if self.phase == "place":
            return 0

        player = self.current_player
        opponent = self.get_opponent(player)
        
        if self.players[player]["tokens_on_board"] <= 2:
            self.phase = "end-capture"
            return opponent
        
        if len(self.get_legal_action_ids(player)) == 0:
            self.phase = "end-block"
            return opponent
        
        if self.current_action > self.maximum_actions_before_draw:
            self.phase = "end-draw"
            return 99

        return 0

    def get_opponent(self, player):
        if player == 1:
            return -1

        return 1
    
class MarelleGame():
    def __init__(self, env:MarelleGymEnv, player1, player2):
        self.env = env
        player1.player_id = 1
        player2.player_id = -1
        self.players = {1: player1, -1: player2}
        self.player_names = {}
        if player1 == "human":
            self.player_names[1] = "human 1"
        else:
            self.player_names[1] = player1.__class__.__name__ + " 1"
        if player2 == "human":
            self.player_names[-1] = "human 2"
        else:
            self.player_names[-1] = player2.__class__.__name__ + " 2"
        
        self.current_player = 1
        self.action_count = 0
        self.action_history = []
    
    def play(self, print_board=True, clear_print_outputs=True):
        self.reset()
        while True:
            if clear_print_outputs:
                clear_output()

            if print_board:     
                # Don't highlight actions for the first move
                if self.action_count > 0:
                    action_highlight = self.env.board.id_to_action[self.action_history[-1]]
                else:
                    action_highlight = None

                print(f"{self.player_names[self.current_player]}'s turn to play :")
                self.env.render(action_highlight=action_highlight)

            interrupt, done = self.step()
            
            if interrupt == True:
                print("Game interrupted, run MarelleGame.play() to continue")
                return self.action_history
            if done:
                if print_board:
                    self.env.render()
                    if self.env.board.check_if_end() == self.current_player:
                        print(f"Game ended with {self.player_names[self.env.board.check_if_end()]} as the winner !")
                    else:
                        print("Game ended with a draw")
                return self.action_history
            
            self.current_player *= -1
        
        return self.action_history

    def reset(self):
        self.env.reset()
        self.current_player = 1
        self.action_count = 0
        self.action_history = []


    def step(self) -> (bool, bool):
        '''
        Returns game_interruption(bool), game_finished(bool)
        '''
        game_interrupted = False
        game_finished = False
        if self.players[self.current_player] == "human":
            
            legal_actions = self.env.board.get_legal_actions(self.current_player)
            legal_action_ids = self.env.board.get_legal_action_ids(self.current_player)

            print("Legal moves :")
            for i in range(len(legal_action_ids)):
                print(f"{legal_action_ids[i]} : {legal_actions[i]}")
            
            try:
                action_id = int(input(f"{self.player_names[self.current_player]} to act :"))
            except ValueError:
                print("Incorrect action input type, please enter an int")
                game_interrupted = True
                return True, False
            
            if action_id not in legal_action_ids:
                print("input is illegal action id")
                game_interrupted = True
                return True, False

        else:
            action_id = self.players[self.current_player].learned_act(self.env.board.get_state())

        observation, reward, done, info = self.env.step(action_id)
        self.action_count += 1
        self.action_history.append(action_id)
        return False, done
    
    def evaluate(self, n_games, player_id):
        if self.players[1] == "human" or self.players[-1] == "human":
            raise Exception('Cannot evaluate humans, they are too slow of a specie')
        
        evaluation = {
            "n_actions":  0,
            "n_captures_place":  0,
            "n_captured_place":  0,
            "n_captures_move":  0,
            "n_captured_move":  0,
            "draws_%":  0,
            "victories_capture_%":  0,
            "victories_block_%":  0,
            "defeats_capture_%":  0,         
            "defeats_block_%":  0        
        }
        
        evaluation_loop = tqdm(range(n_games), desc=f"Evaluation as player {player_id}", leave=False)
        for i in evaluation_loop:
            self.reset()
            game_history = self.play(print_board=False, clear_print_outputs=False)

            current_player = 1
            board = MarelleBoard()

            n_actions = 0
            for action_id in game_history:
                n_actions += 1
                token, capture = board.id_to_action[action_id]
                if capture != None:
                    if current_player == player_id:
                        evaluation["n_captures_" + board.phase] += 1
                    else:
                        evaluation["n_captured_" + board.phase] += 1
            

                board.play_action(action_id)

                winner = board.check_if_end()
                if winner == 99:
                    evaluation["draws_%"] += 1
                    break
                elif winner != 0:
                    if winner == player_id:
                        victory_string = "victories"
                    else:
                        victory_string = "defeats"
                    if board.phase == "end-capture":
                        victory_type = "capture"
                    elif board.phase == "end-block":
                        victory_type = "block"
                    
                    evaluation[f"{victory_string}_{victory_type}_%"] += 1
                    break

                current_player *= -1
            
            evaluation["n_actions"] += n_actions
            
        for key in evaluation:
            evaluation[key] /= n_games

        return evaluation

        

            

