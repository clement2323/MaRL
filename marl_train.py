from marl_agents import ReinforceAgent, MarelleAgent
from marl_env import MarelleGymEnv
import wandb

def train_agent(
    env: MarelleGymEnv,
    n_epochs: int,
    n_trajectories: int,
    trained_agent: ReinforceAgent,
    opponent_agent: MarelleAgent,
    evaluate_agent: MarelleAgent,
    log_training: bool,
    save_model_freq: int,
    evaluate_freq: int):

    if log_training:
        run = wandb.init(project="marl")
        wandb.config.n_trajectories = n_trajectories
        wandb.n_epochs = n_epochs

        # Trained agent
        wandb.config.trained_agent = trained_agent.__class__.__name__
        wandb.config.win_reward = trained_agent.win_reward
        wandb.config.defeat_reward = trained_agent.defeat_reward
        wandb.config.capture_reward = trained_agent.capture_reward
        wandb.config.captured_reward = trained_agent.captured_reward
        wandb.config.lr = trained_agent.lr
        wandb.config.optimizer_name = trained_agent.optimizer.__class__.__name__
        
        # Opponent agent
        wandb.config.opponent_agent = opponent_agent.__class__.__name__

        # Evaluation
        wandb.config.evaluation_agent = evaluate_agent.__class__.__name__

        # Models to watch metrics from
        if trained_agent.model != None
            wandb.watch(trained_agent.model)
        
        # TODO - add other models when added
        
    trained_agent.train(n_trajectories, n_epochs, opponent_agent, evaluate_agent, log_training, save_model_freq, evaluate_freq)

    if log_training:
        run.finish()
    