from marl_env import MarelleGame


def evaluate(env, agent, evaluation_agent, n_games, agent_player_id):
    game = MarelleGame(env, agent, evaluation_agent)
    return game.evaluate(n_games, agent_player_id)
