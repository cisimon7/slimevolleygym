import os
import time
from shutil import copyfile

import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.ppo import PPO

import slimevolleygym

BASE_MODEL = PPO.load("PPO_SelfPlay/best_model.zip")  # Load model to use as base model
INCREASE_PERIOD = 1e4
MAX_SEED: int = 100
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(1e2)
TIME_STEPS = int(1e9)

LOG_DIR = "PPO_TrainingTour11"

# Models to be trained against initial states
models_archive = []


class TrainAgainstArchiveAgentsEnv(slimevolleygym.SlimeVolleyEnv):
    def __init__(self):
        super(TrainAgainstArchiveAgentsEnv, self).__init__()
        self.policy = self
        self.best_model = None
        self.best_model_filename = None

    def predict(self, obs):
        if self.best_model is None:
            return self.action_space.sample()
        else:
            action, _ = self.best_model.predict(obs)
            return action

    def step(self, action, other_action=None):
        return super(TrainAgainstArchiveAgentsEnv, self).step(action, other_action)

    def reset(self):
        # load model if it's there
        model_list = [f for f in os.listdir(LOG_DIR) if f.startswith("Agent_")]
        model_list.sort()
        if len(model_list) > 0:
            filename = os.path.join(LOG_DIR, model_list[-1])  # the latest best model
            if filename != self.best_model_filename:  # check current best is not the same as main best
                self.best_model_filename = filename  # save name of current best as main best
                if self.best_model is not None:
                    del self.best_model
                self.best_model = PPO.load(filename, env=self)  # load new main as best
        return super(TrainAgainstArchiveAgentsEnv, self).reset()


class AgainstArchiveAgentCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(AgainstArchiveAgentCallback, self).__init__(*args, **kwargs)
        self.prev_best = self.best_mean_reward  # Variable to hold previous best_mean_reward
        self.threshold = -4.5  # Minimum reward threshold before archive increases

    def _on_step(self) -> bool:
        result = super(AgainstArchiveAgentCallback, self)._on_step()

        # Comparing Current best mean with previous best mean
        # If current is better than previous, then we add new agent to the archive
        if self.best_mean_reward > self.threshold:
            backup_file = os.path.join(
                LOG_DIR, "Agent_" + str(len(models_archive)) + "*" + str(self.best_mean_reward).zfill(5) + ".zip")
            source_file = os.path.join(LOG_DIR, f"best_model.zip")
            copyfile(source_file, backup_file)

            # Add current model state to archive, though archive is not being used in this experiment
            models_archive.append((PPO.load(source_file), self.last_mean_reward))
            print(f"New Agent Variant Added. Agent Count {len(models_archive)}")
            self.threshold = self.best_mean_reward + 0.02  # Increase threshold to add new model to archive

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


if __name__ == '__main__':

    start = time.perf_counter()

    train()

    # Save top 5 models in archive
    for (i, (model, _)) in enumerate(list(sorted(models_archive, key=(lambda archive: archive[1]), reverse=True))[:5]):
        model.save(os.path.join(LOG_DIR, "agent_" + str(i).zfill(3)))

    end = time.perf_counter()

    print(f"Total Training time: {end - start} seconds\nBest Five models saved")
