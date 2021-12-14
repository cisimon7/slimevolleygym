import time
from shutil import copyfile
from typing import Dict, Tuple, Any, Optional, List
import os
import random

from stable_baselines3.common.monitor import Monitor

import slimevolleygym
import concurrent.futures
import multiprocessing as mp
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import Logger, configure

from slimevolleygym import BaselinePolicy

BASE_MODEL = PPO.load("PPO_SelfPlay/best_model.zip")  # Load model to use as base model
INCREASE_PERIOD = 1e5
MAX_SEED: int = 100
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(1e2)
TIME_STEPS = int(1e9)

LOG_DIR = "PPO_TrainingTour"

# Models to be trained against initial states
models_archive: List[Tuple[Optional[BaseAlgorithm], Any]] = [
    # (Model, mean_reward)
    (BASE_MODEL, 0),  # Initial Policy
    (BaselinePolicy(), 0),  # Initial Policy
]


class TrainAgainstArchiveAgentsEnv(slimevolleygym.SlimeVolleyEnv):
    def __init__(self):
        super(TrainAgainstArchiveAgentsEnv, self).__init__()
        self.policy = self

    def predict(self, obs):
        rnd_model, _ = random.choice(models_archive)  # Selects random model from archive of models
        action, _ = rnd_model.predict(obs)
        return action


class AgainstArchiveAgentCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(AgainstArchiveAgentCallback, self).__init__(*args, **kwargs)
        self.current_best = self.best_mean_reward
        self.threshold = 0.1

    def _on_step(self) -> bool:
        result = super(AgainstArchiveAgentCallback, self)._on_step()

        if self.num_timesteps % EVAL_FREQ == 0:
            print(f"Current Best: {self.best_mean_reward}")

        if result and self.current_best < self.best_mean_reward < 0:
            self.model.save(os.path.join(LOG_DIR, "current_best" + str(int(self.best_mean_reward)).zfill(5)))

        # Archive grows only when we've reach certain threshold
        if self.best_mean_reward > self.threshold:
            # Check if number of rounds has increased by INCREASE_PERIOD
            if result and (self.num_timesteps % INCREASE_PERIOD == 0):
                models_archive.append((self.model, self.last_mean_reward))
                print(f"New Agent Variant Added. Agent Count {len(models_archive)}")

            print("TOUR_PLAY: New Positive reward achieved", )
            source_file = os.path.join(LOG_DIR, f"best_model.zip")
            backup_file = os.path.join(LOG_DIR, "Agent_" + str(self.best_mean_reward).zfill(5) + ".zip")
            copyfile(source_file, backup_file)
            self.threshold += 0.1

        return result


def train():
    configure(folder=LOG_DIR)

    env = Monitor(TrainAgainstArchiveAgentsEnv(), LOG_DIR)
    # SubprocVecEnv([(lambda: TrainAgainstAllAgentsEnv()) for _ in range(2)])  # TrainAgainstAllAgentsEnv()
    env.seed(random.choice(range(MAX_SEED)))

    new_model = PPO("MlpPolicy", env, learning_rate=(lambda rate_left: rate_left * 5e-4), verbose=2)
    # (lambda rate_left: rate_left * 3e-4)

    eval_callback = AgainstArchiveAgentCallback(eval_env=env,
                                                best_model_save_path=LOG_DIR,
                                                log_path=LOG_DIR,
                                                eval_freq=EVAL_FREQ,
                                                n_eval_episodes=EVAL_EPISODES,
                                                deterministic=False)

    new_model.learn(total_timesteps=TIME_STEPS, callback=[eval_callback])

    new_model.save(os.path.join(LOG_DIR, "final_model"))  # probably never get to this point.

    env.close()


def do_something(seconds):
    time.sleep(seconds)
    print(f"Done sleeping {seconds}")


if __name__ == '__main__':

    start = time.perf_counter()

    train()

    # Save top 5 models in archive
    for (i, (model, _)) in enumerate(list(sorted(models_archive, key=(lambda archive: archive[1]), reverse=True))[:5]):
        model.save(os.path.join(LOG_DIR, "agent_" + str(i).zfill(3)))

    end = time.perf_counter()

    print(f"Total Training time: {end - start} seconds\nBest Five models saved")
