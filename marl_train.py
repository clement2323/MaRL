from marl_agents import ReinforceAgent, MarelleAgent
from marl_env import MarelleGymEnv
from tqdm import tqdm_notebook as tqdm
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
        wandb.config.model_name = trained_agent.get_model_name()
        
        # Opponent agent
        wandb.config.opponent_agent = opponent_agent.__class__.__name__

        # Evaluation
        wandb.config.evaluation_agent = evaluate_agent.__class__.__name__

        # Models to watch metrics from
        if trained_agent.model != None:
            wandb.watch(trained_agent.model, log='None', log_freq=5)
        
        
        # TODO - add other models when added
        
    trained_agent.train(n_trajectories, n_epochs, opponent_agent, evaluate_agent, log_training, save_model_freq, evaluate_freq)

    if log_training:
        run.finish()
    
def adversarial_training(
    env: MarelleGymEnv,
    n_trajectories: int,
    first_agent: ReinforceAgent,
    second_agent: ReinforceAgent,      
    evaluate_agent: MarelleAgent,
    log_training: bool,
    save_model_freq: int,
    n_optimize_steps_per_agent = 10,
    n_epochs = 500):


    epoch_loop = tqdm(range(n_epochs), desc="Both agent epoch")
    for i in epoch_loop:
        env.reset()

        train_agent(
            env=env,
            n_epochs=n_optimize_steps_per_agent,
            n_trajectories=n_trajectories,
            trained_agent=first_agent,
            opponent_agent=second_agent,
            evaluate_agent=evaluate_agent,
            log_training=log_training,
            save_model_freq= save_model_freq,
            evaluate_freq= n_optimize_steps_per_agent
        )
        env.reset()
        
        train_agent(
            env=env,
            n_epochs=n_optimize_steps_per_agent,
            n_trajectories=n_trajectories,
            trained_agent=second_agent,
            opponent_agent=first_agent,
            evaluate_agent=evaluate_agent,
            log_training=log_training,
            save_model_freq= save_model_freq,
            evaluate_freq= n_optimize_steps_per_agent
        )
