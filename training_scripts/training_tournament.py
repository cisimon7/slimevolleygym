from typing import Dict, Tuple, Any
import os
import random
import slimevolleygym
import multiprocessing as mp
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import Logger, configure

BASE_MODEL = PPO.load("")  # Load model to use as base model
REWARD_DIFF = 0.5  # must achieve a mean score above this to replace prev best self
TRAINING_UNITS: int = 3  # Should be depending on machine being run on
NUM_ROUNDS: int = 10
MAX_SEED: int = 100
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(1e2)
TIME_STEPS = int(1e4)

LOG_DIR = "training_tour"

# Models to be trained against initial states
models_and_scores: Dict[int, Tuple[Any, float]] = {
    0: (BASE_MODEL, 0.0),
    1: (BASE_MODEL, 0.0),
    2: (BASE_MODEL, 0.0),
    3: (BASE_MODEL, 0.0),
    4: (BASE_MODEL, 0.0)
}


class AgainstAgentEnv(slimevolleygym.SlimeVolleyEnv):
    def __init__(self, other_model):
        super(AgainstAgentEnv, self).__init__()
        self.policy = self
        self.model = other_model

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action


class AgainstAgentCallback(EvalCallback):
    def __init__(self, key: int, *args, **kwargs):
        super(AgainstAgentCallback, self).__init__(*args, **kwargs)
        self.best_mean_reward = models_and_scores[key][1]
        self.generation = 0
        self.key = key

    # Check if new_model has trained enough to surpass current best model
    def _on_step(self) -> bool:
        result = super(AgainstAgentCallback, self)._on_step()
        if result and self.last_mean_reward > (self.best_mean_reward + REWARD_DIFF):
            print(f"Agent {self.key} mean_reward surpassed")

        return result


# Train against the best and replace the worse in the list
def train(agent_key, agent_model):
    configure(folder=LOG_DIR)

    env = AgainstAgentEnv(agent_model)
    env.seed(random.choice(range(MAX_SEED)))

    new_model = PPO("MlpPolicy", env, clip_range=0.2, ent_coef=0.0, n_epochs=10, n_steps=2048, batch_size=64,
                    gamma=0.99, learning_rate=3e-4, verbose=2)

    eval_callback = AgainstAgentCallback(key=agent_key,
                                         eval_env=env,
                                         best_model_save_path=LOG_DIR,
                                         log_path=LOG_DIR,
                                         eval_freq=EVAL_FREQ,
                                         n_eval_episodes=EVAL_EPISODES,
                                         deterministic=False)

    new_model.learn(total_timesteps=TIME_STEPS, callback=eval_callback)

    # Remove the worst model and replace it with new_model
    worst_key, worst_value = sorted(models_and_scores.items(), key=lambda entry: entry[1][1])[-1]
    models_and_scores.pop(worst_key)
    models_and_scores[worst_key + 1] = (new_model, 0.0)

    env.close()


if __name__ == '__main__':
    assert (TRAINING_UNITS < len(models_and_scores))
    for _ in range(NUM_ROUNDS):
        # Pick the three best models
        for (key, (model, mean_reward)) in sorted(models_and_scores, key=lambda entry: entry[1][1])[:TRAINING_UNITS]:
            # Train against the three best models
            train(key, model)

    # Save top 3 models
    for (key, (model, _)) in models_and_scores[:3]:
        model.save(os.path.join(LOG_DIR, "final_model" + str(key).zfill(5)))
