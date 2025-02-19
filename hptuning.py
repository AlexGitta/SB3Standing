import mujoco
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
import matplotlib.pyplot as plt
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from config.config import *
from main import StandupEnv
import gym
from gym import spaces

def objective(trial):
    env = DummyVecEnv([lambda: StandupEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    n_epochs = trial.suggest_int('n_epochs', 5, 50)
    ent_coef = trial.suggest_float('ent_coef', 0, 1e-1)
    vf_coef = trial.suggest_float('vf_coef', 0.5, 1.0)

    model = PPO('MlpPolicy', env, learning_rate=learning_rate, n_epochs=n_epochs, ent_coef=ent_coef, vf_coef=vf_coef, verbose=1, clip_range=0.2)
    checkpoint_callback = CheckpointCallback(save_freq=SAVE_AT_STEP, save_path='./checkpoints/', name_prefix='ppo_model')

    model.learn(total_timesteps=MAX_STEPS, callback=checkpoint_callback)

    # Evaluate the model and return the mean reward
    mean_reward = evaluate_model(model, env)
    return mean_reward

def evaluate_model(model, env, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / num_episodes

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)

    print("Best hyperparameters: ", study.best_params)

if __name__ == "__main__":
    main()