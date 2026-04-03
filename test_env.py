from quadruped_env import QuadrupedWalkEnv

env = QuadrupedWalkEnv(xml_path="Models/quadruped.xml")

obs, info = env.reset()

print("obs shape:", obs.shape)
print("action space:", env.action_space)
print("observation space:", env.observation_space)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        print("Episode ended early")
        obs, _ = env.reset()

print("Done random rollout")