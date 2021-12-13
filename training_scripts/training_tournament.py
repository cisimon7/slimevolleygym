import time
from shutil import copyfile
from typing import Dict, Tuple, Any, Optional, List
import os
import random

import slimevolleygym
import concurrent.futures
import multiprocessing as mp
from stable_baselines3.ppo import PPO
from multiprocessing import Value, Lock, Process
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import Logger, configure

from slimevolleygym import BaselinePolicy

BASE_MODEL = PPO.load("PPO_SelfPlay/best_model.zip")  # Load model to use as base model
REWARD_DIFF = 0.5  # must achieve a mean score above this to replace prev best self
TRAINING_UNITS: int = 3  # Should be depending on machine being run on
INCREASE_PERIOD = 10
NUM_ROUNDS: int = 10
MAX_SEED: int = 100
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(1e2)
TIME_STEPS = int(1e4)

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

        if self.num_timesteps % EVAL_FREQ:
            print(f"Current Best: {self.best_mean_reward}")

        if result and self.current_best < self.best_mean_reward < 0:
            self.model.save(os.path.join(LOG_DIR, "current_best"+str(int(self.best_mean_reward)).zfill(5)))

        # Check if number of rounds has increased by INCREASE_PERIOD
        if result and (self.evaluations_timesteps % INCREASE_PERIOD == 0):
            models_archive.append((self.model, self.last_mean_reward))
            print("New Agent Variant Added to archive")

        if self.best_mean_reward > self.threshold:
            print("TOUR_PLAY: New Positive reward achieved", )
            source_file = os.path.join(LOG_DIR, f"best_model.zip")
            backup_file = os.path.join(LOG_DIR, "Agent_" + str(self.best_mean_reward).zfill(5) + ".zip")
            copyfile(source_file, backup_file)
            self.threshold += 0.1

        return result


def train():
    configure(folder=LOG_DIR)

    env = TrainAgainstArchiveAgentsEnv()
    # SubprocVecEnv([(lambda: TrainAgainstAllAgentsEnv()) for _ in range(2)])  # TrainAgainstAllAgentsEnv()
    env.seed(random.choice(range(MAX_SEED)))

    new_model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=2)

    eval_callback = AgainstArchiveAgentCallback(eval_env=env,
                                                best_model_save_path=LOG_DIR,
                                                log_path=LOG_DIR,
                                                eval_freq=EVAL_FREQ,
                                                n_eval_episodes=EVAL_EPISODES,
                                                deterministic=False)

    new_model.learn(total_timesteps=TIME_STEPS, callback=[eval_callback])

    model.save(os.path.join(LOG_DIR, "final_model"))  # probably never get to this point.

    env.close()


def do_something(seconds):
    time.sleep(seconds)
    print(f"Done sleeping {seconds}")


if __name__ == '__main__':

    start = time.perf_counter()

    train()

    # processes = []
    # for round in range(NUM_ROUNDS):
    #     for (model, _) in models_archive:
    #         p = Process(target=train, args=(round,))
    #         p.start()
    #         processes.append(p)
    #
    #     for process in processes:
    #         process.join()

    # for round in range(NUM_ROUNDS):
    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         executor.map(
    #             train,  # Train to beat current best
    #             [round for _ in range(len(models_archive))]
    #         )

    # Save top 5 models in archive
    for (i, (model, _)) in enumerate(list(sorted(models_archive, key=(lambda archive: archive[1]), reverse=True))[:5]):
        model.save(os.path.join(LOG_DIR, "agent_" + str(i).zfill(3)))

    end = time.perf_counter()

    print(f"Total Training time: {end - start} seconds\nBest Five models saved")
