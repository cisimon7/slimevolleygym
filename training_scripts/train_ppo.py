#!/usr/bin/env python3

# Train single CPU PPO1 on slimevolley.
# Should solve it (beat existing AI on average over 1000 trials) in 3 hours on single CPU, within 3M steps.

import os
import gym
import slimevolleygym
from slimevolleygym import SurvivalRewardEnv

from stable_baselines3 import PPO
# from stable_baselines.ppo1 import PPO1
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.logger import configure
# from stable_baselines3 import logger
from stable_baselines3.common.callbacks import EvalCallback

NUM_TIMESTEPS = int(2e7)
SEED = 721
EVAL_FREQ = 250000
EVAL_EPISODES = 1000
LOGDIR = "ppo"  # moved to zoo afterwards.

# logger.configure(folder=LOGDIR)
logger = configure(folder=LOGDIR)

env = gym.make("SlimeVolley-v0")
env.seed(SEED)

# take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
# model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=4096, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
#                  optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', verbose=2)

model = PPO("MlpPolicy", env, clip_range=0.2, ent_coef=0.0, n_epochs=10,
            n_steps=2048, batch_size=64, gamma=0.99, learning_rate=3e-4, verbose=2)
model.set_logger(logger)

eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ,
                             n_eval_episodes=EVAL_EPISODES)

model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

model.save(os.path.join(LOGDIR, "final_model3"))  # probably never get to this point.

env.close()
