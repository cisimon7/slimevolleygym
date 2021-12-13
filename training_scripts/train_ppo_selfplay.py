#!/usr/bin/env python3

# Simple self-play PPO trainer

import os
import gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

import slimevolleygym

# from stable_baselines import logger
# from stable_baselines.ppo1 import PPO1
from stable_baselines3.ppo import PPO, MlpPolicy
from stable_baselines3.common.monitor import Monitor
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines.common.callbacks import EvalCallback
from stable_baselines3.common.logger import Logger, configure

from shutil import copyfile  # keep track of generations

# Settings
SEED = 17
NUM_TIMESTEPS = int(1e9)  # 1e9
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(1e2)
BEST_THRESHOLD = 0.5  # must achieve a mean score above this to replace prev best self

RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.

# LOGDIR = "ppo1_selfplay"
LOGDIR = "PPO_SelfPlay"


# Is ths a gym environment or a policy? Seems to me like a combination of both
# A type of SlimVolleyEnv and also can be used as a Policy class with predict
# Since step is not overridden, then the loaded prev best model only trains with the BaselinePolicy()
class SlimeVolleySelfPlayEnv(slimevolleygym.SlimeVolleyEnv):
    # wrapper over the normal single player env, but loads the best self play model
    def __init__(self):
        super(SlimeVolleySelfPlayEnv, self).__init__()
        self.policy = self  # acting as a policy class. Here, it overrides the BaselinePolicy in the SlimeVolleyEnv
        self.best_model = None
        self.best_model_filename = None

    def predict(self, obs):  # the policy
        if self.best_model is None:
            return self.action_space.sample()  # return a random action
        else:
            action, _ = self.best_model.predict(obs)
            return action

    # Reset brings the best previously trained policy
    def reset(self):
        # load model if it's there
        model_list = [f for f in os.listdir(LOGDIR) if f.startswith("history")]
        model_list.sort()
        if len(model_list) > 0:
            filename = os.path.join(LOGDIR, model_list[-1])  # the latest best model
            if filename != self.best_model_filename:  # check current best is not the same as main best
                print("loading model: ", filename)
                self.best_model_filename = filename  # save current best as main best
                if self.best_model is not None:
                    del self.best_model
                self.best_model = PPO.load(filename, env=self)  # load new main best
        return super(SlimeVolleySelfPlayEnv, self).reset()


class SelfPlayCallback(EvalCallback):
    # hacked it to only save new version of best model if beats prev self by BEST_THRESHOLD score
    # after saving model, resets the best score to be BEST_THRESHOLD
    def __init__(self, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        # self.best_mean_reward = BEST_THRESHOLD
        self.generation = 0

    def _on_step(self) -> bool:
        result = super(SelfPlayCallback, self)._on_step()

        if self.num_timesteps % 10000 == 0:
            print(f"Current Best: {self.last_mean_reward}")

        if result and self.best_mean_reward > BEST_THRESHOLD:
            self.generation += 1
            print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
            print("SELFPLAY: new best model, bumping up generation to", self.generation)
            source_file = os.path.join(LOGDIR, f"best_model.zip")
            backup_file = os.path.join(LOGDIR, "history_" + str(self.generation).zfill(8) + ".zip")
            copyfile(source_file, backup_file)
            self.best_mean_reward = BEST_THRESHOLD
        return result


def rollout(env, policy):
    """ play one agent vs the other in modified gym-style loop. """
    obs = env.reset()

    done = False
    total_reward = 0

    while not done:

        action, _states = policy.predict(obs)
        obs, reward, done, _ = env.step(action)  # Doesn't seem to be playing against a modified version of itself

        total_reward += reward

        if RENDER_MODE:
            env.render()

    return total_reward


def train():
    # train selfplay agent
    # logger.configure(folder=LOGDIR)
    configure(folder=LOGDIR)

    env = SubprocVecEnv([(lambda: Monitor(env=SlimeVolleySelfPlayEnv(), filename=LOGDIR)) for _ in range(2)])
    env.seed(SEED)

    # take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
    # model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=4096, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
    #              optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', verbose=2)
    model = PPO(MlpPolicy, env, clip_range=0.2, ent_coef=0.0, n_epochs=5, n_steps=1024, batch_size=64, gamma=0.99,
                gae_lambda=0.95, learning_rate=(lambda rate: rate * 0.0001), verbose=2)

    eval_callback = SelfPlayCallback(env,
                                     best_model_save_path=LOGDIR,
                                     log_path=LOGDIR,
                                     eval_freq=EVAL_FREQ,
                                     n_eval_episodes=EVAL_EPISODES,
                                     deterministic=False)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model"))  # probably never get to this point.

    env.close()


if __name__ == "__main__":
    train()
