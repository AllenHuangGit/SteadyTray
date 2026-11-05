"""Common observation processing utilities for deployment."""

import numpy as np
from collections import deque
import torch
from typing import Tuple, Optional
from scripts.config import Config

def get_encoder_obs_dim(num_actions: int, object_obs_size: int = 6) -> int:
    """
    Get the dimension of student encoder observations.
    
    Args:
        num_actions: Number of actions (29 for G1)
        object_obs_size: Size of object observations (default: 6 for pos+rpy)
        
    Returns:
        Total encoder observation dimension: 96 + object_obs_size
    """
    base_obs_dim = 3 + 3 + 3 + num_actions + num_actions + num_actions  # 96 for G1
    return base_obs_dim + object_obs_size


# Helper function to compute gravity orientation from quaternion
def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def detect_policy_type(policy: torch.jit.ScriptModule) -> str:
    """
    Detect whether the policy is standard (single input) or distillation (dual input).
    
    Args:
        policy: Loaded torch.jit policy network
        
    Returns:
        "standard" or "distillation"
    """
    try:
        # Method 1: Check state_dict for distillation-specific components
        state_dict = policy.state_dict()
        
        # Distillation policies have student_encoder and actor_body components
        if "student_encoder.embed.weight" in state_dict:
            print("Detected distillation policy: found student_encoder in state_dict")
            return "distillation"
        
        if "actor_body.0.base.weight" in state_dict:
            print("Detected distillation policy: found actor_body (FiLM adapters) in state_dict")
            return "distillation"
        
        if "frozen_actor.0.weight" in state_dict or "residual_adapter.residual_mlp.0.weight" in state_dict:
            print("Detected distillation policy: found frozen_actor/residual_adapter (Residual) in state_dict")
            return "distillation"
        
        # Method 2: Try to get the forward method signature
        # Distillation policies have forward(student_encoder_obs, policy_obs)
        # Standard policies have forward(observations)
        try:
            code = policy.code
            if "student_encoder_obs" in code or "policy_obs" in code:
                print("Detected distillation policy: found dual inputs in code")
                return "distillation"
        except Exception:
            pass
        
        # Method 3: Check the number of inputs by inspecting the graph
        try:
            graph = policy.graph
            graph_str = str(graph)
            
            # Count input nodes (excluding self)
            if "student_encoder_obs" in graph_str:
                print("Detected distillation policy: found student_encoder_obs in graph")
                return "distillation"
            
            # Distillation policies have 3 inputs: self + 2 observation tensors
            # Standard policies have 2 inputs: self + 1 observation tensor
            inputs = list(graph.inputs())
            if len(inputs) > 2:
                print(f"Detected distillation policy: found {len(inputs)} inputs (>2) in graph")
                return "distillation"
        except Exception:
            pass
        
        print("Detected standard policy: no distillation components found")
        return "standard"
        
    except Exception as e:
        # Default to standard if detection fails
        print(f"Policy type detection failed: {e}, defaulting to standard")
        return "standard"


def detect_encoder_obs_size(policy: torch.jit.ScriptModule) -> int:
    """
    Automatically detect the encoder observation size from a distillation policy.
    
    This inspects the policy's state_dict to find the student_encoder.embed.weight shape,
    similar to how batch_processing.py detects dimensions. The embedding input dimension
    corresponds to the encoder observation size.
    
    Args:
        policy: Loaded torch.jit policy network (should be distillation type)
        
    Returns:
        Encoder observation dimension (e.g., 102 for 6D object obs)
        Falls back to 102 (96 base + 6 object) if detection fails
    """
    try:
        # Method 1: Try to get state_dict and inspect student_encoder.embed.weight
        # This is the same approach as batch_processing.py
        state_dict = policy.state_dict()
        
        # Look for student_encoder embedding weight
        # Shape is [embed_dim, input_dim] where input_dim is what we want
        if "student_encoder.embed.weight" in state_dict:
            embed_weight = state_dict["student_encoder.embed.weight"]
            student_embed_dim = embed_weight.shape[1]  # Input dimension
            print(f"Auto-detected encoder observation size from state_dict: {student_embed_dim}")
            return student_embed_dim
            
    except Exception as e:
        print(f"Could not auto-detect from state_dict: {e}")
        
    # Default fallback: 96 (base) + 6 (pos + projected_gravity)
    default_size = 96 + 6
    print(f"Using default encoder observation size: {default_size}")
    return default_size


def compute_policy_action(
    policy: torch.jit.ScriptModule,
    frame_stack: deque,
    qj: np.ndarray,
    dqj: np.ndarray,
    quat: np.ndarray,
    omega: np.ndarray,
    cmd: np.ndarray,
    previous_action: np.ndarray,
    config: Config,
    object_obs: Optional[np.ndarray] = None,
    policy_type: Optional[str] = None,
    encoder_frame_stack: Optional[deque] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process sensor observations and compute policy action.
    
    This function combines observation normalization, frame stacking, policy inference,
    and action transformation. It can be used by both MuJoCo simulation and real robot deployment.
    Supports both standard policies and distillation (adapter student) policies.
    
    Args:
        policy: Loaded torch.jit policy network
        frame_stack: Deque with maxlen=5 containing observation history for standard policy
        qj: Raw joint positions (robot order)
        dqj: Raw joint velocities (robot order)
        quat: Quaternion [w, x, y, z] or [x, y, z, w] depending on get_gravity_orientation
        omega: Raw angular velocity (3,)
        cmd: Command values (3,) - typically [vel_x, vel_y, yaw_rate]
        previous_action: Previous action in robot order (num_actions,)
        config: Configuration object with necessary parameters
        object_obs: Optional object observations for distillation policies 
                   Flexible size with first 3 dims as position, rest varies:
                   - [pos(3), projected_gravity(3)] = 6 dims (default, no orientation)
                   - [pos(3), quat(4)] = 7 dims (position + quaternion)
                   - [pos(3), rot6d(6)] = 9 dims (position + 6D rotation)
                   - [pos(3), rotmat(9)] = 12 dims (position + rotation matrix)
        policy_type: Optional policy type ("standard" or "distillation"). If None, auto-detect.
        encoder_frame_stack: Optional deque with maxlen=32 for encoder observation history
                           Required for distillation policies. Encoder obs size will be
                           96 + object_obs_size (default: 96 + 6 = 102 dims per timestep).
        
    Returns:
        action_robot: Action in robot order (num_actions,)
        target_dof_pos: Target joint positions in robot order (num_actions,)
    """
    
    # Auto-detect policy type if not provided
    if policy_type is None:
        policy_type = detect_policy_type(policy)

    default_angles = config.default_angles[config.policy_to_robot]

    # Normalize observations (vectorized operations)
    qj_normalized = (qj - default_angles) * config.dof_pos_scale
    dqj_normalized = dqj * config.dof_vel_scale
    gravity_orientation = get_gravity_orientation(quat)
    omega_normalized = omega * config.ang_vel_scale

    # Process observation with frame stacking (group-major order)
    big_group_major = process_observation_with_frame_stack(
        frame_stack=frame_stack,
        omega=omega_normalized,
        gravity_orientation=gravity_orientation,
        cmd=cmd,
        qj=qj_normalized[config.robot_to_policy],
        dqj=dqj_normalized[config.robot_to_policy],
        action=previous_action[config.robot_to_policy],
        cmd_scale=config.cmd_scale,
        num_obs=config.num_obs,
        num_actions=config.num_actions
    )
    
    # Policy inference - different handling for standard vs distillation
    with torch.no_grad():
        if policy_type == "distillation":
            # Distillation policy requires two inputs:
            # 1. student_encoder_obs: Observation history [seq_len=32, batch=1, obs_dim=96+object_size]
            # 2. policy_obs: Full observation history (5-frame stack) [batch=1, policy_obs_dim=480]
            
            # encoder_frame_stack should be provided by caller (initialized in deployment scripts)
            assert encoder_frame_stack is not None, "encoder_frame_stack must be provided for distillation policies"
            
            # Build current timestep encoder observation
            current_encoder_obs = build_student_encoder_obs(
                omega=omega_normalized,
                gravity_orientation=gravity_orientation,
                cmd=cmd,
                qj=qj_normalized[config.robot_to_policy],
                dqj=dqj_normalized[config.robot_to_policy],
                action=previous_action[config.robot_to_policy],
                cmd_scale=config.cmd_scale,
                num_actions=config.num_actions,
                object_obs=object_obs
            )
            
            # Add current observation to encoder history
            encoder_frame_stack.append(current_encoder_obs)
            
            # Convert encoder frame stack to tensor with shape [seq_len, batch, obs_dim]
            # deque -> numpy array [32, obs_dim] -> tensor [32, 1, obs_dim]
            # where obs_dim = 96 + object_obs_size (default: 102)
            encoder_obs_array = np.array(encoder_frame_stack, dtype=np.float32)  # Shape: [32, obs_dim]
            student_encoder_tensor = torch.from_numpy(encoder_obs_array).unsqueeze(1)  # Shape: [32, 1, obs_dim]
            
            # Convert policy observations to tensor [batch, policy_obs_dim]
            policy_obs_tensor = torch.from_numpy(big_group_major).unsqueeze(0)  # Shape: [1, 480]
            
            # Call distillation policy with two inputs
            # student_encoder_tensor: [32, 1, obs_dim], policy_obs_tensor: [1, 480]
            action_policy = policy(student_encoder_tensor, policy_obs_tensor).squeeze(0).numpy()
        else:
            # Standard policy - single input
            obs_tensor = torch.from_numpy(big_group_major).unsqueeze(0)
            action_policy = policy(obs_tensor).squeeze(0).numpy()
    
    # Convert action from policy order to robot order
    action_robot = action_policy[config.policy_to_robot]

    # Transform action to target joint positions (fused multiply-add)
    target_dof_pos = action_robot * config.action_scale[config.policy_to_robot] + default_angles

    return action_robot, target_dof_pos


def build_student_encoder_obs(
    omega: np.ndarray,
    gravity_orientation: np.ndarray,
    cmd: np.ndarray,
    qj: np.ndarray,
    dqj: np.ndarray,
    action: np.ndarray,
    cmd_scale: np.ndarray,
    num_actions: int,
    object_obs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Build student encoder observations for a single timestep.
    
    For distillation policies, the student encoder receives observations that match
    the StudentEncoderCfg structure:
    - base_ang_vel (3)
    - projected_gravity (3)
    - velocity_commands (3)
    - joint_pos_rel (29)
    - joint_vel_rel (29)
    - last_action (29)
    - object_pos_cam (3) - object position
    - object_data_cam (variable) - object data in various formats:
      * 3 dims: projected_gravity (default 6D format)
      * 4 dims: quaternion (7D format)
      * 6 dims: 6D rotation (9D format)
      * 9 dims: rotation matrix (12D format)
    
    Total: 96 + object_obs_size dimensions per timestep.
    Default object_obs_size = 6 (pos + projected_gravity) gives 102 total dimensions.
    
    Args:
        omega: Angular velocity (3,) - normalized (corresponds to base_ang_vel)
        gravity_orientation: Gravity vector in body frame (3,) (corresponds to projected_gravity)
        cmd: Command values (3,) - typically [vel_x, vel_y, yaw_rate] (corresponds to velocity_commands)
        qj: Joint positions (num_actions,) - normalized, policy order (corresponds to joint_pos_rel)
        dqj: Joint velocities (num_actions,) - normalized, policy order (corresponds to joint_vel_rel)
        action: Previous action (num_actions,) - policy order (corresponds to last_action)
        cmd_scale: Scale factors for commands (3,)
        num_actions: Number of actions (29 for G1)
        object_obs: Optional object observations with flexible size
                   - First 3 elements: object position [x, y, z]
                   - Remaining elements: varies by format
                   - Common formats:
                     * 6D: [pos(3), projected_gravity(3)] = 6 dims (default, no orientation)
                     * 7D: [pos(3), quat(4)] = 7 dims (position + quaternion)
                     * 9D: [pos(3), rot6d(6)] = 9 dims (position + 6D rotation)
                     * 12D: [pos(3), rotmat(9)] = 12 dims (position + rotation matrix)
                   If not provided, must raise error for distillation policies.
        
    Returns:
        student_encoder_obs: Student encoder observations for one timestep
                            Shape: (obs_dim,) where obs_dim = 96 + object_obs_size
    """
    # Base proprioceptive observations: omega(3) + gravity(3) + cmd(3) + qj(29) + dqj(29) + action(29) = 96
    base_obs_dim = 3 + 3 + 3 + num_actions + num_actions + num_actions
    
    # Determine object observation size
    if object_obs is not None and len(object_obs) > 0:
        object_obs_size = len(object_obs)
        # Ensure we have at least position (3 dims)
        if object_obs_size < 3:
            print(f"Warning: object_obs size {object_obs_size} < 3, padding with zeros")
            object_data = np.zeros(6, dtype=np.float32)  # Default to 6 dims
            object_data[:object_obs_size] = object_obs
            object_obs_size = 6
        else:
            object_data = object_obs.astype(np.float32)
    else:
        raise ValueError("object_obs must be provided for distillation policies")
    
    # Total dimension per timestep: base_obs + object_obs_size
    total_dim = base_obs_dim + object_obs_size
    student_obs = np.zeros(total_dim, dtype=np.float32)
    
    # Fill in observations following StudentEncoderCfg order
    idx = 0
    
    # base_ang_vel (3)
    student_obs[idx:idx+3] = omega
    idx += 3
    
    # projected_gravity (3)
    student_obs[idx:idx+3] = gravity_orientation
    idx += 3
    
    # velocity_commands (3)
    student_obs[idx:idx+3] = cmd * cmd_scale
    idx += 3
    
    # joint_pos_rel (29)
    student_obs[idx:idx+num_actions] = qj
    idx += num_actions
    
    # joint_vel_rel (29)
    student_obs[idx:idx+num_actions] = dqj
    idx += num_actions
    
    # last_action (29)
    student_obs[idx:idx+num_actions] = action
    idx += num_actions
    
    # object_pos_cam (first 3 dims of object_obs)
    student_obs[idx:idx+3] = object_data[:3]
    idx += 3
    
    # object_data_cam (remaining dims of object_obs)
    # This can be projected_gravity (3), quaternion (4), rot6d (6), or rotmat (9)
    data_size = object_obs_size - 3
    if data_size > 0:
        student_obs[idx:idx+data_size] = object_data[3:3+data_size]
        idx += data_size
    
    return student_obs


def process_observation_with_frame_stack(
    frame_stack: deque,
    omega: np.ndarray,
    gravity_orientation: np.ndarray,
    cmd: np.ndarray,
    qj: np.ndarray,
    dqj: np.ndarray,
    action: np.ndarray,
    cmd_scale: np.ndarray,
    num_obs: int,
    num_actions: int
) -> np.ndarray:
    """
    Process observations with frame stacking and group-major ordering.
    
    This function builds a single frame observation, adds it to the frame stack,
    and then restructures all frames into group-major order (all omega across frames,
    then all gravity, etc.) as expected by the policy network.
    
    Args:
        frame_stack: Deque with maxlen=5 containing observation history
        omega: Angular velocity (3,)
        gravity_orientation: Gravity vector in body frame (3,)
        cmd: Command values (3,) - typically [vel_x, vel_y, yaw_rate]
        qj: Joint positions (num_actions,) - already normalized/scaled
        dqj: Joint velocities (num_actions,) - already normalized/scaled
        action: Previous action (num_actions,)
        cmd_scale: Scale factors for commands (3,)
        num_obs: Observation dimension per frame (typically 96)
        num_actions: Number of actions (typically 29)
        
    Returns:
        big_group_major: Stacked observations in group-major order (5 * num_obs,)
                        Total dimensions: 5 frames × num_obs = 480 for 29-DOF
    """
    # Create temporary observation buffer
    obs = np.zeros(num_obs, dtype=np.float32)
    
    # Build single frame observation (num_obs dimensions: 3+3+3+29+29+29 = 96)
    # Use slicing for faster assignment
    obs[:3] = omega
    obs[3:6] = gravity_orientation
    obs[6:9] = cmd * cmd_scale
    obs[9 : 9 + num_actions] = qj
    obs[9 + num_actions : 9 + 2 * num_actions] = dqj
    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
    
    # Add current frame to frame stack
    frame_stack.append(obs)
    
    # Convert deque to array once (faster than multiple operations)
    stacked_obs = np.array(frame_stack, dtype=np.float32)  # Shape: (5, num_obs)
    
    # Extract features using direct slicing (much faster than reshape chains)
    # Shape is already (5, num_obs), so we can directly slice columns
    obs_omega = stacked_obs[:, :3].ravel()  # ravel() is faster than reshape(-1)
    obs_gravity_orientation = stacked_obs[:, 3:6].ravel()
    obs_cmd = stacked_obs[:, 6:9].ravel()
    obs_pos = stacked_obs[:, 9:9 + num_actions].ravel()
    obs_vel = stacked_obs[:, 9 + num_actions : 9 + 2 * num_actions].ravel()
    obs_action = stacked_obs[:, 9 + 2 * num_actions : 9 + 3 * num_actions].ravel()
    
    # Concatenate all features in group-major order
    # Pre-allocate output array for better performance
    total_size = 3*5 + 3*5 + 3*5 + num_actions*5*3  # 15+15+15+435 = 480
    big_group_major = np.empty(total_size, dtype=np.float32)
    
    # Use direct assignment instead of concatenate (faster)
    idx = 0
    big_group_major[idx:idx+15] = obs_omega
    idx += 15
    big_group_major[idx:idx+15] = obs_gravity_orientation
    idx += 15
    big_group_major[idx:idx+15] = obs_cmd
    idx += 15
    big_group_major[idx:idx+num_actions*5] = obs_pos
    idx += num_actions*5
    big_group_major[idx:idx+num_actions*5] = obs_vel
    idx += num_actions*5
    big_group_major[idx:idx+num_actions*5] = obs_action
    
    return big_group_major
