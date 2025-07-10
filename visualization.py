import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C  # noqa: F401

from env import CustomEnv, NUM_OBSERVATION_BINS

MODEL_TIME = "1752161400"
MODEL_ITER = "1370000"

filepath = f"models/{MODEL_TIME}/{MODEL_ITER}.zip"

window_size = 12
bins = NUM_OBSERVATION_BINS


actions = np.zeros((window_size, bins))


env = CustomEnv()
model = A2C("MlpPolicy", env, verbose=1)
model.load(filepath, env)

for i, window_size in enumerate(range(window_size)):
    for j, bin in enumerate(range(bins)):
        obs = np.array([window_size, bin])
        action, _ = model.predict(obs, deterministic=True)
        actions[j, i] = action

print(np.matrix(actions))

plt.imshow(actions, extent=[0, window_size, 0, bins], origin="lower", aspect="auto")
plt.colorbar(label="Action")
plt.xlabel("Window Size")
plt.ylabel("Bins")
plt.title("Policy Action Heatmap")
plt.show()
