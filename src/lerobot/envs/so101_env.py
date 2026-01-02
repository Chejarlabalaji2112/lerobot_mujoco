import gymnasium as gym
import numpy as np
from gymnasium import spaces
# Import the robot class we defined above
from lerobot.robots.so101_mujoco import SO101MuJoCoRobot, SO101MuJoCoConfig
from lerobot.cameras import OpenCVCameraConfig 

class SO101MuJoCoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, scene_xml_path="scene.xml", episode_length=500):
        self.episode_length = episode_length
        self.step_count = 0
        
        # Configure the internal simulated robot
        # We use dummy OpenCV configs just to hold resolution settings
        self.config = SO101MuJoCoConfig(
            scene_xml_path=scene_xml_path,
            cameras={
                "front_cam": OpenCVCameraConfig(width=640, height=480, fps=30),
                "left_cam": OpenCVCameraConfig(width=640, height=480, fps=30)
            },
            sim_fps=30
        )
        self.robot = SO101MuJoCoRobot(self.config)
        self.robot.connect()
        
        # Define Spaces matching LeRobot normalization
        # Observations: Cameras + Agent State
        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict({
                "front_cam": spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                "left_cam": spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8)
            }),
            # State vector: 6 motors
            "agent_pos": spaces.Box(-100, 100, (6,), dtype=np.float32)
        })
        
        # Actions: 6 motors, normalized [-100, 100]
        self.action_space = spaces.Box(-100, 100, (6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.robot.connect() # Resets physics
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        
        # Convert numpy array action back to dict for the Robot class
        # (Assuming action order matches self.config.motors order)
        action_dict = {
            f"{name}.pos": val 
            for name, val in zip(self.config.motors, action)
        }
        
        self.robot.send_action(action_dict)
        
        obs = self._get_obs()
        
        # Basic termination logic
        done = False
        truncated = self.step_count >= self.episode_length
        info = {"is_success": False} # Implement success check logic here if needed
        
        # Dummy reward
        reward = 0.0
        
        return obs, reward, done, truncated, info

    def render(self):
        """Required by lerobot-eval for video saving."""
        # Use front_cam for visualization
        # We need to manually invoke the renderer if we want visual-only return without step
        # But efficiently, we can just grab the last frame from the robot if available
        # Here we force a render call:
        self.robot.renderers["front_cam"].update_scene(self.robot.data, camera="front_cam")
        return self.robot.renderers["front_cam"].render()

    def _get_obs(self):
        raw_obs = self.robot.get_observation()
        
        # Format dictionary for LeRobot Policy consumption
        return {
            "pixels": {
                "front_cam": raw_obs["front_cam"],
                "left_cam": raw_obs["left_cam"]
            },
            # Flatten dict to array
            "agent_pos": np.array([raw_obs[f"{m}.pos"] for m in self.config.motors], dtype=np.float32)
        }
        
    def close(self):
        self.robot.disconnect()