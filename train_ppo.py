from stable_baselines3 import PPO
from quadruped_env import QuadrupedWalkEnv

env = QuadrupedWalkEnv(xml_path="Models/quadruped.xml")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,   # default is 10
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)

model.learn(total_timesteps=200_000)

model.save("ppo_quadruped")