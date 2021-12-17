import os
import time
import random
from shutil import copyfile

import torch
import matplotlib.pyplot as plt
from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

import slimevolleygym
from slimevolleygym import BaselinePolicy

BASE_MODEL = PPO.load("PPO_SelfPlay/best_model.zip")  # Load model to use as base model
INCREASE_PERIOD = 1e4
MAX_SEED: int = 100
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(1e2)
TIME_STEPS = int(1e9)

LOG_DIR = "PPO_TrainingTour4"

# Models to be trained against initial states
models_archive = []


class TrainAgainstArchiveAgentsEnv(slimevolleygym.SlimeVolleyEnv):
    def __init__(self):
        super(TrainAgainstArchiveAgentsEnv, self).__init__()
        self.policy = self

    def predict(self, obs):
        if len(models_archive) == 0:
            return self.action_space.sample()
        else:
            rnd_model, _ = random.choice(models_archive)  # Selects random model from archive of models
            action, _ = rnd_model.predict(obs)
            return action

    def step(self, action, other_action=None):
        return super(TrainAgainstArchiveAgentsEnv, self).step(action, other_action)

    def reset(self):
        return super(TrainAgainstArchiveAgentsEnv, self).reset()


class AgainstArchiveAgentCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(AgainstArchiveAgentCallback, self).__init__(*args, **kwargs)
        self.prev_best = self.best_mean_reward
        self.threshold = -4.5
        self.pretrained = False

    def _on_step(self) -> bool:
        result = super(AgainstArchiveAgentCallback, self)._on_step()

        if self.best_mean_reward > self.threshold:
            backup_file = os.path.join(
                LOG_DIR, "Agent_" + str(len(models_archive)) + "*" + str(self.best_mean_reward).zfill(5) + ".zip")
            source_file = os.path.join(LOG_DIR, f"best_model.zip")
            copyfile(source_file, backup_file)

            models_archive.append((self.model, self.last_mean_reward))
            print(f"New Agent Variant Added. Agent Count {len(models_archive)}")
            self.threshold = self.best_mean_reward + 0.02

        # We add our initial baseline policy and Fixed BaselinePolicy when mean reward becomes positive
        # We add them now because the robot is trained enough to beat them
        if (not self.pretrained) and self.best_mean_reward > (-1):
            self.pretrained = True
            models_archive.append((BASE_MODEL, 0))
            models_archive.append((BaselinePolicy(), 0))

        return result


class PlottingCallback(BaseCallback):
    def __init__(self):
        super(PlottingCallback, self).__init__()
        self.plot = None

    def _on_step(self) -> bool:
        # Get monitor data
        x, y = ts2xy(load_results(LOG_DIR), "timesteps")

        if self.plot is None:
            plt.ion()
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            line, = ax.plot(x, y)
            self.plot = (line, ax, fig)
            plt.show()
        else:
            self.plot[0].set_data(x, y)
            self.plot[1].relim()
            self.plot[1].set_xlim([
                self.locals["total_timesteps"] * -0.02,
                self.locals["total_timesteps"] * +1.02
            ])
            self.plot[1].autoscale_view(True, True, True)
            self.plot[-1].canvas.draw()

        return super(PlottingCallback, self)._on_step()


def train():
    logger = configure(folder=LOG_DIR, format_strings=["stdout", "csv", "tensorboard"])

    torch.set_num_threads(4)

    env = Monitor(TrainAgainstArchiveAgentsEnv(), LOG_DIR)

    env.seed(17)  # random.choice(range(MAX_SEED))

    new_model = PPO("MlpPolicy", env, learning_rate=(lambda rate_left: rate_left * 1e-4), n_steps=1024,
                    batch_size=1024, verbose=2, gae_lambda=0.95, gamma=0.99, ent_coef=0.0)
    new_model.set_logger(logger)

    eval_callback = AgainstArchiveAgentCallback(eval_env=env,
                                                best_model_save_path=LOG_DIR,
                                                log_path=LOG_DIR,
                                                eval_freq=EVAL_FREQ,
                                                n_eval_episodes=EVAL_EPISODES,
                                                deterministic=False)

    plot_callback = PlottingCallback()

    # adding plot_callback makes training run slow
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
