import networkx as nx
import gym
from gym import spaces
from IPython.display import clear_output
from termcolor import colored
import numpy as np

class MarelleGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(MarelleGymEnv, self).__init__()    
    
    # Define action and observation space
    # They must be gym.spaces objects
        self.board = MarelleBoard()
        self.current_player = 1
        
    def step(self, action): 
   
        self.board.play_action(action,self.current_player)
        
        observation=self.board.get_state()
        done = self.board.check_if_end(self.current_player) != 0
        reward = {}
        reward["game_end"] = self.board.check_if_end(self.current_player) * self.current_player # equal to 1 or 0
        
        _, opponent_pos = self.board.id_to_action[action]
        if opponent_pos != None:
            reward["capture_token"] = 1
        else:
            reward["capture_token"] = 0
        
        if self.board.phase == 'move': #good si tu captures beaucoup tu as plus de jeton 
            reward["end_place_phase"] = np.sum(self.board.get_state())
        else:
            reward["end_place_phase"] = 0
            
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
        self.initialize_game()

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

    def initialize_game(self):
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

        self.phase = "place"
        self.players = {1: {}, -1: {}}
        self.players[1]["played_tokens"] = 0
        self.players[-1]["played_tokens"] = 0
        self.players[1]["tokens_on_board"] = 0
        self.players[-1]["tokens_on_board"] = 0

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

    def play_action(self, action_id, player):
        if self.phase == "place":
            self.place_token_action(self.id_to_action[action_id], player)
            
        else:
            self.move_token_action(self.id_to_action[action_id], player)
        self.change_phase_if_needed()
        self.check_if_end(player)
    
    def get_legal_actions(self, player):
        legal_actions = []

        if self.phase == "place":
            legal_actions = self.place_token_legal_actions(player)
        
        elif self.phase == "move":
            legal_actions = self.move_token_legal_actions(player)

        return legal_actions
    
    def get_legal_action_ids(self, player):
        legal_actions = []
        legal_action_ids = []

        if self.phase == "place":
            legal_actions = self.place_token_legal_actions(player)
        
        elif self.phase == "move":
            legal_actions = self.move_token_legal_actions(player)
        
        for action in legal_actions:
            legal_action_ids.append(self.action_to_id[action])

        return legal_action_ids

    def get_state(self):
        state = []
        for node in self.graph.nodes:
            state.append(self.graph.nodes[node]["state"])

        return state
    
    def place_token_action(self, action, player):
        legal_moves = self.place_token_legal_actions(player)

        if action not in legal_moves:
            raise Exception('Illegal place move')

        (new_token_pos, opponent_token_pos) = action
        self.graph.nodes[new_token_pos]["state"] = player
        self.players[player]["played_tokens"] += 1
        self.players[player]["tokens_on_board"] += 1


        if opponent_token_pos != None:
            self.graph.nodes[opponent_token_pos]["state"] = 0
            self.players[self.get_opponent(player)]["tokens_on_board"] -= 1
    
    def place_token_legal_actions(self, player):
        opponent_nodes = []
        for node in self.graph.nodes:
            if self.graph.nodes[node]["state"] == self.get_opponent(player):
                opponent_nodes.append(node)

        legal_actions = []
        for node in self.graph.nodes:
            if self.graph.nodes[node]["state"] == 0:
                self.graph.nodes[node]["state"] = player
                if self.check_if_capture(self.graph.nodes, node, player):
                    for opponent_node in opponent_nodes:
                        legal_actions.append((node, opponent_node))
                else:
                    legal_actions.append((node, None))
                self.graph.nodes[node]["state"] = 0
        
        return legal_actions
    
    def move_token_action(self, action, player):
        edge, opponent_token = action
        legal_moves = self.move_token_legal_actions(player)
        if action not in legal_moves:
            raise Exception('Illegal token move action')

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
            if self.check_if_capture(self.graph.nodes, new_node, player):
                for opponent_node in opponent_nodes:
                    legal_actions.append((edge, opponent_node))
            else:
                legal_actions.append((edge, None))
            self.graph.nodes[new_node]["state"] = 0
            self.graph.nodes[cur_node]["state"] = player
            
        return legal_actions
    
    def check_if_capture(self, nodes, position, player):
        r, t = position
        capture = False

        # step 1, horizontal (same t, different r)
        if (t % 2 == 0):
            r_capture = True
            r_lines = [0, 1, 2]
            for line_r in r_lines:
                if nodes[line_r, t]["state"] != player:
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
                if nodes[r, line_t]["state"] != player:
                    t_capture = False
                    break
            
            if t_capture == True:
                return True

        return False
    
    def change_phase_if_needed(self):
        if self.players[1]["played_tokens"] == self.N_TOKENS_PER_PLAYER and self.players[-1]["played_tokens"] == self.N_TOKENS_PER_PLAYER:
            self.phase = "move"

    def check_if_end(self, player):
        '''
        Returns 0 if the game is not ended and the id of the player if a player won
        '''
        if self.phase == "place":
            return 0

        opponent = self.get_opponent(player)
        if self.players[opponent]["tokens_on_board"] <= 2:
            self.phase = "end-capture"
            return player
        
        if len(self.get_legal_action_ids(opponent)) == 0:
            self.phase = "end-block"
            return player

        return 0

    def get_opponent(self, player):
        if player == 1:
            return -1

        return 1
    
class MarelleGame():
    def __init__(self, env, player1, player2):
        self.env = env
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

            interrupt = self.step()
            
            if interrupt == True:
                print("Game interrupted, run MarelleGame.play() to continue")
                return self.action_history
            if self.env.board.check_if_end(self.current_player) != 0:
                if print_board:
                    self.env.render()
                    print(f"Game ended with {self.player_names[self.env.board.check_if_end(self.current_player)]} as the winner !")
                return self.action_history
            
            self.current_player *= -1
        
        return self.action_history

    def reset(self):
        self.env.reset()
        self.current_player = 1
        self.action_count = 0
        self.action_history = []


    def step(self):
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
                return True
            
            if action_id not in legal_action_ids:
                print("input is illegal action id")
                return True

        else:
            action_id = self.players[self.current_player].learned_act(self.env.board.get_state())

        self.env.board.play_action(action_id, self.current_player)
        self.action_count += 1
        self.action_history.append(action_id)
        return False
    
    def evaluate(self, n_games, player_id):
        if self.players[1] == "human" or self.players[-1] == "human":
            raise Exception('Cannot evaluate humans, they are too slow')
        
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
        
        for i in range(n_games):
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
            

                board.play_action(action_id, current_player)

                winner = board.check_if_end(current_player)
                if winner != 0:
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
                
                if n_actions > 250:
                    evaluation["draws_%"] += 1
                    break

                current_player *= -1
            
            evaluation["n_actions"] += n_actions
            
        for key in evaluation:
            evaluation[key] /= n_games

        return evaluation

        

            

