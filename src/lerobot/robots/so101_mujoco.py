import numpy as np
import mujoco
from dataclasses import dataclass, field
from typing import Any
import logging
from queue import Queue

# Import LeRobot base classes
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.cameras import CameraConfig
from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig

# Import pynput for keyboard
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# =============================================================================
# 1. SIMULTANEOUS KEYBOARD TELEOPERATOR
# =============================================================================

@TeleoperatorConfig.register_subclass("so101_keyboard")
@dataclass
class SO101KeyboardTeleopConfig(TeleoperatorConfig):
    motors: list[str] = field(default_factory=lambda: [
        "shoulder_pan", "shoulder_lift", "elbow_flex", 
        "wrist_flex", "wrist_roll", "gripper"
    ])
    sensitivity: float = 3.0

class SO101KeyboardTeleop(Teleoperator):
    config_class = SO101KeyboardTeleopConfig
    name = "so101_keyboard"

    def __init__(self, config: SO101KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.listener = None
        
        # Track currently held keys
        self.pressed_keys = set()
        
        # Initialize targets
        self.targets = {f"{m}.pos": 0.0 for m in config.motors}

        # Define Key Mapping (Modify this to your preference)
        # Format: 'key': (motor_index, direction)
        self.key_map = {
            # Motor 1: Shoulder Pan
            '1': (0, 1),  'q': (0, -1),
            # Motor 2: Shoulder Lift
            '2': (1, 1),  'w': (1, -1),
            # Motor 3: Elbow
            '3': (2, 1),  'e': (2, -1),
            # Motor 4: Wrist Flex
            '4': (3, 1),  'r': (3, -1),
            # Motor 5: Wrist Roll
            '5': (4, 1),  't': (4, -1),
            # Motor 6: Gripper
            '6': (5, 1),  'y': (5, -1),
        }

    @property
    def is_connected(self):
        return PYNPUT_AVAILABLE and self.listener is not None

    def connect(self):
        if not PYNPUT_AVAILABLE:
            raise ImportError("Install pynput: pip install pynput")
        
        print("\n" + "="*50)
        print(" MULTI-MOTOR CONTROL ACTIVE")
        print(" --------------------------------------------------")
        print(" Controls (Up / Down):")
        print(" [1/Q] Pan      [2/W] Lift     [3/E] Elbow")
        print(" [4/R] W-Flex   [5/T] W-Roll   [6/Y] Gripper")
        print(" [0]   Reset All")
        print("="*50 + "\n")
        
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def disconnect(self):
        if self.listener:
            self.listener.stop()

    def _on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.pressed_keys.add(key.char.lower())
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.pressed_keys.discard(key.char.lower())
        except AttributeError:
            pass

    def get_action(self) -> dict[str, Any]:
        # Reset Logic
        if '0' in self.pressed_keys:
            for k in self.targets:
                self.targets[k] = 0.0
            return self.targets.copy()

        # Update targets based on ALL currently held keys
        for key_char in self.pressed_keys:
            if key_char in self.key_map:
                motor_idx, direction = self.key_map[key_char]
                
                # Safety check
                if motor_idx < len(self.config.motors):
                    motor_name = self.config.motors[motor_idx]
                    key_name = f"{motor_name}.pos"
                    
                    # Update Position
                    self.targets[key_name] += (self.config.sensitivity * direction)
                    
                    # Clamp
                    self.targets[key_name] = np.clip(self.targets[key_name], -100.0, 100.0)

        return self.targets.copy()

    @property
    def action_features(self):
        return {f"{m}.pos": float for m in self.config.motors}
        
    @property
    def feedback_features(self):
        return {}

    def calibrate(self): pass
    def configure(self): pass
    def send_feedback(self, feedback): pass
    def is_calibrated(self): return True


# =============================================================================
# 2. MUJOCO ROBOT WRAPPER (Unchanged from previous working version)
# =============================================================================

@RobotConfig.register_subclass("so101_mujoco")
@dataclass
class SO101MuJoCoConfig(RobotConfig):
    scene_xml_path: str = "scene.xml"
    motors: list[str] = field(default_factory=lambda: [
        "shoulder_pan", "shoulder_lift", "elbow_flex", 
        "wrist_flex", "wrist_roll", "gripper"
    ])
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    sim_fps: int = 30

class SO101MuJoCoRobot(Robot):
    config_class = SO101MuJoCoConfig
    name = "so101_mujoco"

    def __init__(self, config: SO101MuJoCoConfig):
        super().__init__(config)
        self.config = config
        
        print(f"Loading MuJoCo: {config.scene_xml_path}")
        self.model = mujoco.MjModel.from_xml_path(config.scene_xml_path)
        self.data = mujoco.MjData(self.model)
        
        self._motor_map = {}
        for name in config.motors:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id == -1:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id != -1:
                    for i in range(self.model.nu):
                        if self.model.actuator_trnid[i, 0] == joint_id:
                            actuator_id = i
                            break
            
            if actuator_id == -1:
                print(f"Warning: Motor {name} not found in XML.")
                continue

            joint_id = self.model.actuator_trnid[actuator_id, 0]
            limit_min, limit_max = self.model.jnt_range[joint_id]
            if limit_max <= limit_min: limit_min, limit_max = -3.14, 3.14

            self._motor_map[name] = {
                "act_id": actuator_id,
                "qpos_addr": self.model.jnt_qposadr[joint_id],
                "min": limit_min, 
                "max": limit_max
            }

        self.cameras = {
            k: mujoco.Renderer(self.model, height=v.height, width=v.width) 
            for k, v in self.config.cameras.items()
        }
        self.logs = {}
        self._is_connected = False

    @property
    def is_connected(self): return self._is_connected

    def connect(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._is_connected = True

    def disconnect(self): self._is_connected = False

    def get_observation(self) -> dict[str, Any]:
        obs = {}
        for name, info in self._motor_map.items():
            raw = self.data.qpos[info["qpos_addr"]]
            rng = info["max"] - info["min"]
            norm = ((raw - info["min"]) / rng) * 200.0 - 100.0
            
            obs[f"{name}.pos"] = float(norm)
            self.logs[name] = float(norm)

        for k, renderer in self.cameras.items():
            renderer.update_scene(self.data, camera=k)
            obs[k] = renderer.render()
            
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        for name, info in self._motor_map.items():
            if f"{name}.pos" in action and action[f"{name}.pos"] is not None:
                tgt_norm = action[f"{name}.pos"]
                tgt_norm = np.clip(tgt_norm, -100.0, 100.0)
                rng = info["max"] - info["min"]
                tgt_rad = ((tgt_norm + 100.0) / 200.0) * rng + info["min"]
                self.data.ctrl[info["act_id"]] = tgt_rad

        steps = int((1.0 / self.config.sim_fps) / self.model.opt.timestep)
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            
        return action

    @property
    def observation_features(self):
        f = {f"{m}.pos": float for m in self.config.motors}
        for c, cfg in self.config.cameras.items(): f[c] = (cfg.height, cfg.width, 3)
        return f

    @property
    def action_features(self):
        return {f"{m}.pos": float for m in self.config.motors}

    @property
    def is_calibrated(self): return True
    def calibrate(self): pass
    def configure(self): pass