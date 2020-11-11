from marl_env import MarelleGame
from marl_agents import BetterRandomAgent


def evaluate_against_betterrandomagent(env, n_games, agent, agent_player_id):
    evaluation_agent = BetterRandomAgent(env, agent_player_id * -1)
    return evaluate(env, agent, evaluation_agent, n_games, agent_player_id)


def evaluate(env, agent, evaluation_agent, n_games, agent_player_id):
    game = MarelleGame(env, agent, evaluation_agent)
    return game.evaluate(n_games, agent_player_id)
