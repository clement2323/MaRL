import networkx as nx
import copy

class MarelleBoard():
    '''
    player 1 versus player -1
    3 phases : place, move and end
    initialize_game() => Reset the board
    print_board() => render the board
    action_list_by_id => dict id : action
    action_id_by_action => dict action : id
    play_action(action_id) => play an action
    get_legal_actions() => get list of action ids that are legal
    check_if_end() => returns 0 if not ended, 1 or -1 if a player won
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

        action_list_by_id = {}
        action_id_by_action = {}
        i = 0
        for action in self.place_token_action_list:
            action_list_by_id[i] = action
            action_id_by_action[action] = i
            i += 1

        for action in self.move_token_action_list:
            action_list_by_id[i] = action
            action_id_by_action[action] = i
            i += 1
        
        self.action_list_by_id = action_list_by_id
        self.action_id_by_action = action_id_by_action

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



    def print_board(self):
        v = {}
        for node in self.graph.nodes:
            if self.graph.nodes[node]["state"] == -1:
                v[node] = 2
            else:
                v[node] = self.graph.nodes[node]["state"]


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
            self.place_token_action(self.action_list_by_id[action_id], player)
            
        else:
            self.move_token_action(self.action_list_by_id[action_id], player)
        self.change_phase_if_needed()
        self.check_if_end()
    
    def get_legal_action_ids(self, player):
        legal_action_ids = []

        if self.phase == "place":
            legal_actions = self.place_token_legal_actions()
        
        elif self.phase == "move":
            legal_actions = self.move_token_legal_actions()
        
        for action in legal_actions:
            legal_action_ids.append(self.action_id_by_action[action])

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
                temp_graph = copy.deepcopy(self.graph)
                temp_graph.nodes[node]["state"] = player
                if self.check_if_capture(temp_graph.nodes, node, player):
                    for opponent_node in opponent_nodes:
                        legal_actions.append((node, opponent_node))
                else:
                    legal_actions.append((node, None))
        
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
            temp_graph = copy.deepcopy(self.graph)
            temp_graph.nodes[new_node]["state"] = player
            temp_graph.nodes[cur_node]["state"] = 0
            if self.check_if_capture(temp_graph.nodes, new_node, player):
                for opponent_node in opponent_nodes:
                    legal_actions.append((edge, opponent_node))
            else:
                legal_actions.append((edge, None))
            
        return legal_actions
    
    def check_if_capture(self, nodes, position, player):
        r, t = position
        capture = False

        # step 1, horizontal (same t, different r)
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
        if self.players[1]["played_tokens"] == self.N_TOKENS_PER_PLAYER and self.players[-1]["played_tokens"] == N_TOKENS_PER_PLAYER:
            self.phase = "move"

    def check_if_end(self):
        '''
        Returns 0 if the game is not ended and the id of the player if a player won
        '''
        if self.phase != "move":
            return 0

        if self.players[1]["tokens_on_board"] == 2:
            print("Game ended with player -1 as winner")
            self.phase = "end"
            return -1
        elif self.players[-1]["tokens_on_board"] == 2:
            print("Game ended with player 1 as winner")
            self.phase = "end"
            return 1
        return 0

    def get_opponent(self, player):
        if player == 1:
            return -1

        return 1