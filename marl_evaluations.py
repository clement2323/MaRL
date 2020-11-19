from marl_env import MarelleGame


def evaluate(env, evaluated_agent, evaluation_agent, n_games):
    init_agent_player_id = evaluated_agent.player_id
    init_opponent_player_id = evaluation_agent.player_id

    game = MarelleGame(env, evaluated_agent, evaluation_agent)

    evaluated_agent.player_id = 1
    evaluation_agent.player_id = -1
    evaluation_starting_first = game.evaluate(n_games, evaluated_agent.player_id)

    evaluated_agent.player_id = -1
    evaluation_agent.player_id = 1

    game = MarelleGame(env, evaluation_agent, evaluated_agent)
    evaluation_starting_second = game.evaluate(n_games, evaluated_agent.player_id)

    final_evaluation = {}
    for key in evaluation_starting_first:
        final_evaluation[key] = round((evaluation_starting_first[key] + evaluation_starting_second[key]) / 2, 3)
    
    evaluated_agent.player_id = init_agent_player_id
    evaluation_agent.player_id = init_opponent_player_id

    return final_evaluation

