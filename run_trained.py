import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from quadruped_env import QuadrupedWalkEnv

env = QuadrupedWalkEnv(xml_path="Models/quadruped.xml")
model = PPO.load("ppo_quadruped", env=env)

obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        viewer.sync()

        if terminated or truncated:
            obs, info = env.reset()