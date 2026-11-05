"""Real-robot deployment of SteadyTray policies on the Unitree G1 (29-DoF).

Usage (from the repository root):

    python deploy/deploy_real/deploy_real.py --net <NET_INTERFACE> \
        --config deploy/configs/g1_29dof_walk.yaml \
        --policy exported/policy_9999.pt

See deploy/deploy_real/README.md for the full startup procedure.

The controller structure (state machine, DDS setup) is adapted from unitree_rl_gym
(https://github.com/unitreerobotics/unitree_rl_gym), Copyright (c) 2016-2023
HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"), BSD 3-Clause License,
see deploy/deploy_real/LICENSE.unitree_rl_gym.
"""

from typing import Optional
import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add deploy/ to the path so the shared `scripts` package can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.policy_runner import compute_policy_action, detect_encoder_obs_size, detect_policy_type
from scripts.config import Config

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from common.remote_controller import RemoteController, KeyMap


# Proprioceptive observation size per timestep: ang_vel(3) + gravity(3) + cmd(3) + 3 * 29 joints
BASE_PROPRIO_DIM = 96

# Ankle pitch / roll joints (robot order) are excluded from the joint-limit check
JOINT_LIMIT_IGNORE = [4, 5, 10, 11]


class Controller:
    def __init__(self, config: Config, base_policy_path: str, residual_policy_path: Optional[str] = None,
                 encoder_seq_len: int = 32, enable_auto_switch: bool = False, max_detection_timeout: float = 0.5,
                 soft_limit_factor: float = 1.7) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Auto-switching parameters
        self.enable_auto_switch = enable_auto_switch
        self.max_detection_timeout = max_detection_timeout

        if enable_auto_switch:
            print(f"Automatic policy switching ENABLED (timeout: {max_detection_timeout}s)")
        else:
            print("Automatic policy switching DISABLED (manual control only)")

        # Initialize the policy networks
        self.base_policy = torch.jit.load(base_policy_path)
        self.base_policy_type = detect_policy_type(self.base_policy)
        print(f"Base policy type: {self.base_policy_type}")

        self.residual_policy = None
        self.residual_policy_type = None
        if residual_policy_path is not None:
            self.residual_policy = torch.jit.load(residual_policy_path)
            self.residual_policy_type = detect_policy_type(self.residual_policy)
            print(f"Residual policy type: {self.residual_policy_type}")

        # Current active policy (can be switched during runtime)
        self.current_policy = self.base_policy
        self.current_policy_type = self.base_policy_type
        self.use_residual = False

        # Button states for rising-edge detection
        self.button_states = {}

        # Process variables (29-DoF whole body)
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.counter = 0

        # Pre-cache config values for faster access in the run loop
        self.num_actions = config.num_actions
        self.control_dt = config.control_dt
        self.vel_x_cmd = config.vel_x_cmd
        self.vel_y_cmd = config.vel_y_cmd
        self.yaw_cmd = config.yaw_cmd
        self.kps = config.kps
        self.kds = config.kds

        # Joint limits (robot order) and the derived soft limits
        self.soft_limits = self.compute_soft_limit(config.joint_limits_lower, config.joint_limits_upper,
                                                   soft_limit_factor)

        # Frame stack for the 5-frame policy observation
        self.frame_stack = deque(maxlen=5)
        for _ in range(5):
            self.frame_stack.append(np.zeros(config.num_obs, dtype=np.float32))

        # Auto-detect the encoder observation size of distillation policies
        encoder_obs_dim = None
        if self.base_policy_type == 'distillation':
            encoder_obs_dim = detect_encoder_obs_size(self.base_policy)
        if self.residual_policy is not None and self.residual_policy_type == 'distillation' and encoder_obs_dim is None:
            encoder_obs_dim = detect_encoder_obs_size(self.residual_policy)

        # encoder_obs_dim = 96 (proprioception) + object_obs_size
        object_obs_size = None
        if encoder_obs_dim is not None:
            object_obs_size = encoder_obs_dim - BASE_PROPRIO_DIM
            print(f"Detected object observation size: {object_obs_size} "
                  f"(position: 3D, orientation: {object_obs_size - 3}D)")

        # AprilTag detector is only needed for distillation (object-aware) policies
        self.apriltag_detector = None
        if object_obs_size is not None:
            # Imported lazily so that standard policies can run without camera dependencies
            from scripts.apriltag_camera import AprilTagDetector
            self.apriltag_detector = AprilTagDetector(object_obs_size=object_obs_size, **config.camera)
            self.apriltag_detector.start()
            time.sleep(2)
            print(f"AprilTag detector initialized with object_obs_size={object_obs_size}")

        # Encoder history for distillation policies
        self.encoder_seq_len = encoder_seq_len
        self.encoder_obs_dim = encoder_obs_dim
        self.encoder_frame_stack = deque(maxlen=encoder_seq_len)
        self.reset_encoder_history()

        # G1 uses the hg message type
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        # Wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command message
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def reset_encoder_history(self):
        """Fill the encoder history with zeros (used at startup and on every policy switch)."""
        if self.encoder_obs_dim is None:
            return
        self.encoder_frame_stack.clear()
        for _ in range(self.encoder_seq_len):
            self.encoder_frame_stack.append(np.zeros(self.encoder_obs_dim, dtype=np.float32))

    def switch_policy(self, use_residual: bool, reason: str):
        print(f"Switch to {'residual' if use_residual else 'base'} policy ({reason})")
        self.use_residual = use_residual
        if use_residual:
            self.current_policy = self.residual_policy
            self.current_policy_type = self.residual_policy_type
        else:
            self.current_policy = self.base_policy
            self.current_policy_type = self.base_policy_type
        # Clear the encoder history to avoid stale observations
        self.reset_encoder_history()

    def is_button_pressed(self, key):
        """Return True only on the rising edge (released -> pressed) of a remote-controller button."""
        current = self.remote_controller.button[key]
        last = self.button_states.get(key, 0)
        self.button_states[key] = current
        return current == 1 and last == 0

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2  # seconds
        num_step = int(total_time / self.control_dt)

        # default_angles is in policy order, kps/kds are in robot order
        default_pos_robot = self.config.default_angles[self.config.policy_to_robot]
        num_motors = self.num_actions

        # Record the current joint positions (robot order)
        init_dof_pos = np.zeros(num_motors, dtype=np.float32)
        motor_states = self.low_state.motor_state
        for i in range(num_motors):
            init_dof_pos[i] = motor_states[i].q

        # Linearly interpolate to the default pose
        motor_cmds = self.low_cmd.motor_cmd
        for step in range(num_step):
            alpha = step / num_step
            for motor_idx in range(num_motors):
                cmd = motor_cmds[motor_idx]
                cmd.q = init_dof_pos[motor_idx] * (1 - alpha) + default_pos_robot[motor_idx] * alpha
                cmd.qd = 0
                cmd.kp = self.kps[motor_idx]
                cmd.kd = self.kds[motor_idx]
                cmd.tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")

        default_pos_robot = self.config.default_angles[self.config.policy_to_robot]
        motor_cmds = self.low_cmd.motor_cmd

        while self.remote_controller.button[KeyMap.A] != 1:
            # Hold all motors at the default pose
            for motor_idx in range(self.num_actions):
                cmd = motor_cmds[motor_idx]
                cmd.q = default_pos_robot[motor_idx]
                cmd.qd = 0
                cmd.kp = self.kps[motor_idx]
                cmd.kd = self.kds[motor_idx]
                cmd.tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    @staticmethod
    def compute_soft_limit(lower_limit, upper_limit, soft_factor):
        """Scale the joint range around its center by `soft_factor`."""
        center = (upper_limit + lower_limit) / 2.0
        half_range = (upper_limit - lower_limit) / 2.0
        return (center - half_range * soft_factor, center + half_range * soft_factor)

    @staticmethod
    def check_joint_limit(target, soft_limit):
        """Return True if every checked joint target is finite and within the soft limits."""
        checked = np.ones(len(target), dtype=bool)
        checked[JOINT_LIMIT_IGNORE] = False

        lower_violations = (target < soft_limit[0]) & checked
        upper_violations = (target > soft_limit[1]) & checked
        non_finite = ~np.isfinite(target)

        if not (lower_violations.any() or upper_violations.any() or non_finite.any()):
            return True

        print("Joint limit violations detected:")
        for idx in np.where(non_finite)[0]:
            print(f"  Joint {idx}: non-finite target {target[idx]}")
        for idx in np.where(lower_violations)[0]:
            print(f"  Joint {idx}: {target[idx]:.4f} < lower_limit {soft_limit[0][idx]:.4f} "
                  f"(violation: {target[idx] - soft_limit[0][idx]:.4f})")
        for idx in np.where(upper_violations)[0]:
            print(f"  Joint {idx}: {target[idx]:.4f} > upper_limit {soft_limit[1][idx]:.4f} "
                  f"(violation: {target[idx] - soft_limit[1][idx]:.4f})")
        return False

    def run(self):
        start_time = time.time()
        self.counter += 1

        # Joint positions and velocities (robot order)
        motor_states = self.low_state.motor_state
        qj = self.qj
        dqj = self.dqj
        for i in range(self.num_actions):
            motor = motor_states[i]
            qj[i] = motor.q
            dqj[i] = motor.dq

        # IMU state
        imu = self.low_state.imu_state
        quat = np.array(imu.quaternion, dtype=np.float32)
        ang_vel = np.array(imu.gyroscope, dtype=np.float32)

        # Velocity command from the remote controller joysticks
        rc = self.remote_controller
        cmd = self.cmd
        ly = rc.ly
        cmd[0] = 0.0 if abs(ly) <= 0.1 else (self.vel_x_cmd[1] if ly > 0 else self.vel_x_cmd[0])
        lx = rc.lx
        cmd[1] = 0.0 if abs(lx) <= 0.1 else (self.vel_y_cmd[1] if lx < 0 else self.vel_y_cmd[0])
        rx = rc.rx
        cmd[2] = 0.0 if abs(rx) <= 0.1 else (self.yaw_cmd[1] if rx < 0 else self.yaw_cmd[0])

        # ------------------------------------------------------------------
        # Policy switching
        #   Manual (always enabled when a residual policy is given):
        #     - R2: switch to the residual policy (only if the AprilTag is visible)
        #     - B:  switch back to the base policy
        #   Automatic (--enable_auto_switch):
        #     - switch back to the base policy when the AprilTag has not been
        #       seen for longer than --max_detection_timeout seconds
        # ------------------------------------------------------------------
        object_obs = None
        can_see_object = False
        if self.apriltag_detector is not None:
            object_obs = self.apriltag_detector.get_object_obs()
            can_see_object = self.apriltag_detector.is_target_detected()

        if self.residual_policy is not None:
            if self.is_button_pressed(KeyMap.R2) and not self.use_residual:
                if can_see_object:
                    self.switch_policy(True, "R2 pressed")
                else:
                    print("Cannot switch to residual policy: object not visible")

            if self.is_button_pressed(KeyMap.B) and self.use_residual:
                self.switch_policy(False, "B pressed")

            if self.enable_auto_switch and self.use_residual and self.apriltag_detector is not None:
                if self.apriltag_detector.get_time_since_last_detection() > self.max_detection_timeout:
                    self.switch_policy(False, "object not visible")

        # Policy inference
        self.action, target_dof_pos = compute_policy_action(
            policy=self.current_policy,
            frame_stack=self.frame_stack,
            qj=qj,
            dqj=dqj,
            quat=quat,
            omega=ang_vel,
            cmd=cmd,
            previous_action=self.action,
            config=self.config,
            object_obs=object_obs,
            policy_type=self.current_policy_type,
            encoder_frame_stack=self.encoder_frame_stack
        )

        # Safety check before sending the targets to the motors
        if not self.check_joint_limit(target_dof_pos, self.soft_limits):
            print("ERROR: Action exceeds joint limits! Emergency stop triggered.")
            return False

        motor_cmds = self.low_cmd.motor_cmd
        for i in range(self.num_actions):
            motor_cmd = motor_cmds[i]
            motor_cmd.q = target_dof_pos[i]
            motor_cmd.qd = 0
            motor_cmd.kp = self.kps[i]
            motor_cmd.kd = self.kds[i]
            motor_cmd.tau = 0
        self.send_cmd(self.low_cmd)

        # Sleep for the remaining time in the control cycle
        elapsed_time = time.time() - start_time
        sleep_time = self.control_dt - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"Warning: Control loop took {elapsed_time:.4f}s, exceeds control_dt {self.control_dt:.4f}s")

        return True

    def shutdown(self):
        """Put the motors in damping mode and release the camera."""
        create_damping_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        if self.apriltag_detector is not None:
            self.apriltag_detector.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SteadyTray real-robot deployment (Unitree G1 29-DoF)")
    parser.add_argument("--net", type=str, required=True,
                        help="Network interface connected to the robot (e.g. enp3s0)")
    parser.add_argument("--config", type=str, default="deploy/configs/g1_29dof_walk.yaml",
                        help="Path to the robot configuration file")
    parser.add_argument("--policy", type=str, default="exported/policy_9999.pt",
                        help="Policy that runs after pressing A")
    parser.add_argument("--residual_policy", type=str, default=None,
                        help="(Optional) policy to switch to with R2 / back with B")
    parser.add_argument("--encoder_seq_len", type=int, default=32,
                        help="Encoder sequence length (number of history frames for distillation policies)")
    parser.add_argument("--enable_auto_switch", action="store_true",
                        help="Automatically switch back to --policy when the AprilTag is lost")
    parser.add_argument("--max_detection_timeout", type=float, default=0.5,
                        help="Seconds without seeing the AprilTag before the automatic switch back")
    parser.add_argument("--soft_limit_factor", type=float, default=1.7,
                        help="Scale of the joint range used by the joint-limit emergency stop "
                             "(1.0 = hardware limits, lower = more conservative)")
    args = parser.parse_args()

    config = Config(args.config)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config, args.policy, args.residual_policy, args.encoder_seq_len,
                            args.enable_auto_switch, args.max_detection_timeout, args.soft_limit_factor)

    try:
        # Zero torque state, press START to continue
        controller.zero_torque_state()
        # Move to the default pose
        controller.move_to_default_pos()
        # Hold the default pose, press A to start the policy
        controller.default_pos_state()

        while True:
            if not controller.run():
                print("Exiting due to joint limit violation")
                break
            # Press SELECT to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Enter the damping state
        controller.shutdown()
        print("Exit")
