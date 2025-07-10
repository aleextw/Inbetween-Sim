import os
import time

from stable_baselines3 import PPO, A2C  # noqa: F401
from stable_baselines3.common.env_checker import check_env

from env import CustomEnv


models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = CustomEnv()
check_env(env)

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10_000
ITERS = 10_000

for iter in range(1, ITERS + 1):
    env.reset()
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS * iter}")
    env.render()
