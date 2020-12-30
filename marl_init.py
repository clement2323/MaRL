import importlib

from marl_env import MarelleBoard, MarelleGymEnv, MarelleGame
from marl_models import FCModel, ConvModel, ConvModel_small_output
from marl_agents import RandomAgent, BetterRandomAgent, SingleModelReinforce, ReinforceAgent, MarelleAgent, TripleModelReinforce
from marl_train import train_agent, adversarial_training
from marl_evaluations import evaluate

import os
import cProfile

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb

import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")