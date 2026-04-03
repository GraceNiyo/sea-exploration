import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class QuadrupedWalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        xml_path: str,
        frame_skip: int = 5,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.render_mode = render_mode

        # 12 muscle actuators in your XML
        self.nu = self.model.nu
        assert self.nu == 12, f"Expected 12 actuators, got {self.nu}"

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.nu,),
            dtype=np.float32,
        )

        # Observation:
        # torso quat (4)
        # torso lin vel (3)
        # torso ang vel (3)
        # 8 joint positions
        # 8 joint velocities
        # 4 foot contacts
        self.obs_dim = 4 + 3 + 3 + 8 + 8 + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Joint names in the order you want them in the observation
        self.joint_names = [
            "rbthigh", "rbshin",
            "rfthigh", "rfshin",
            "lbthigh", "lbshin",
            "lfthigh", "lfshin",
        ]

        self.touch_sensor_names = [
            "rbfoot_touch_sensor",
            "rffoot_touch_sensor",
            "lbfoot_touch_sensor",
            "lffoot_touch_sensor",
        ]

        self.joint_qpos_adr = []
        self.joint_qvel_adr = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_qpos_adr.append(self.model.jnt_qposadr[jid])
            self.joint_qvel_adr.append(self.model.jnt_dofadr[jid])

        self.sensor_adr = []
        for name in self.touch_sensor_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = self.model.sensor_adr[sid]
            self.sensor_adr.append(adr)

        self.sensor_dim = [
            self.model.sensor_dim[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)]
            for name in self.touch_sensor_names
        ]

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        # Free joint torso:
        # qpos[0:3] = x, y, z
        # qpos[3:7] = quaternion
        torso_quat = qpos[3:7].copy()
        torso_linvel = qvel[0:3].copy()
        torso_angvel = qvel[3:6].copy()

        joint_pos = np.array([qpos[i] for i in self.joint_qpos_adr], dtype=np.float32)
        joint_vel = np.array([qvel[i] for i in self.joint_qvel_adr], dtype=np.float32)

        foot_contacts = np.array([
            self.data.sensordata[self.sensor_adr[0]],
            self.data.sensordata[self.sensor_adr[1]],
            self.data.sensordata[self.sensor_adr[2]],
            self.data.sensordata[self.sensor_adr[3]],
        ], dtype=np.float32)

        obs = np.concatenate([
            torso_quat.astype(np.float32),
            torso_linvel.astype(np.float32),
            torso_angvel.astype(np.float32),
            joint_pos,
            joint_vel,
            foot_contacts,
        ]).astype(np.float32)

        return obs

    def _get_forward_velocity(self):
        """
        Forward velocity along the torso's local x-axis.
        This is better than using world x when the robot can yaw.
        """
        quat = self.data.qpos[3:7].copy()
        rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot, quat)
        R = rot.reshape(3, 3)

        world_linvel = self.data.qvel[0:3].copy()
        body_x_world = R[:, 0]  # body x-axis expressed in world coordinates
        forward_vel = float(np.dot(world_linvel, body_x_world))
        return forward_vel

    def _is_terminated(self):
        qpos = self.data.qpos

        # torso height
        z = qpos[2]

        # quaternion sanity
        quat = qpos[3:7]
        if np.any(np.isnan(qpos)) or np.any(np.isnan(self.data.qvel)):
            return True

        # crude fall condition
        if z < 0.20:
            return True

        # optional: if quaternion is corrupted
        if np.linalg.norm(quat) < 1e-6:
            return True

        return False

    def _compute_reward(self, action):
        forward_vel = self._get_forward_velocity()

        # small survival reward
        alive_reward = 0.1

        # control cost
        ctrl_cost = 0.01 * float(np.sum(np.square(action)))

        # torso stability penalty:
        # use angular velocity as a simple first proxy
        angvel = self.data.qvel[3:6]
        stability_penalty = 0.01 * float(np.sum(np.square(angvel)))

        reward = forward_vel + alive_reward - ctrl_cost - stability_penalty
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # small randomized initial conditions
        self.data.qpos[:] = self.model.qpos0.copy()
        self.data.qvel[:] = 0.0

        # optional small noise for robustness
        self.data.qpos[:] += self.np_random.normal(0.0, 0.005, size=self.data.qpos.shape)
        self.data.qvel[:] += self.np_random.normal(0.0, 0.005, size=self.data.qvel.shape)

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = False

        info = {
            "forward_velocity": self._get_forward_velocity(),
            "torso_height": float(self.data.qpos[2]),
            "control_cost": float(0.01 * np.sum(np.square(action))),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        # Minimal placeholder.
        # For actual rendering, add mujoco.Renderer or use a viewer.
        pass

    def close(self):
        pass