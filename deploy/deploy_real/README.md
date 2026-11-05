# Deploy on the Physical Robot (Sim2Real)

Python deployment of SteadyTray policies on the **Unitree G1 (29-DoF)** through `unitree_sdk2py` low-level control. The object pose on the tray is estimated online with a RealSense camera and an AprilTag.

> **Safety first.** Always start with the robot hanging from a gantry / hoist, keep the remote controller in hand, and be ready to press `select` (damping) at any time.

## Hardware

| Component | Notes |
|---|---|
| Unitree G1, 29-DoF | Wrist joints included, on-board locomotion controller disabled (debug mode) |
| Tray holders | Mounted on both hands, meshes in [`deploy/deploy_mujoco/g1_xml/meshes/`](../deploy_mujoco/g1_xml/meshes/) (`tray_holder_side_left.stl`, `tray_holder_side_right.stl`) |
| Intel RealSense D435 | Mounted on the torso, matching the `d435_camera_frame` site in [`g1_tray_holder.xml`](../deploy_mujoco/g1_xml/g1_tray_holder.xml) |
| AprilTag | Printed on the **top face** of the object. Family / ID / size are set in the `camera` block of the config (default `tag16h5`, ID `0`, 50 mm) |
| Host PC | Connected to the robot over Ethernet, camera connected over USB 3 |

## Installation

```bash
cd deploy/deploy_real
conda env create -f environment.yml
conda activate g1_deploy_real
```

Install the [Unitree Python SDK](https://github.com/unitreerobotics/unitree_sdk2_python):

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

If the installation complains that `cyclonedds` cannot be found, build [CycloneDDS 0.10.x](https://github.com/eclipse-cyclonedds/cyclonedds) first and set `CYCLONEDDS_HOME` as described in the SDK README.

## Configuration

All deployment parameters live in [`deploy/configs/g1_29dof_walk.yaml`](../configs/g1_29dof_walk.yaml) (shared with the MuJoCo deployment). The real-robot specific entries are:

```yaml
lowcmd_topic: "rt/lowcmd"
lowstate_topic: "rt/lowstate"

camera:
  families: "tag16h5"     # AprilTag family
  tag_id: 0               # ID of the tag attached to the object
  tag_size: 0.05          # Tag edge length in meters
  resolution: [640, 480]
  fps: 30

vel_x_cmd: [-0.3, 0.5]    # joystick velocity commands (min, max)
vel_y_cmd: [-0.4, 0.4]
yaw_cmd: [-0.4, 0.4]
```

Before running on the robot, check that the camera sees the tag and reports a sensible pose:

```bash
# from the repository root
python deploy/scripts/apriltag_camera.py --config deploy/configs/g1_29dof_walk.yaml
```

## Network Setup

Connect the host PC to the robot with an Ethernet cable and give the interface a static IP on the robot subnet `192.168.123.x`, e.g. `192.168.123.99/24`. Find the interface name with:

```bash
ip addr
```

## Usage

Run from the **repository root**:

```bash
python deploy/deploy_real/deploy_real.py \
    --net <NET_INTERFACE> \
    --config deploy/configs/g1_29dof_walk.yaml \
    --policy exported/policy_9999.pt
```

| Argument | Default | Description |
|---|---|---|
| `--net` | *(required)* | Network interface connected to the robot, e.g. `enp3s0` |
| `--config` | `deploy/configs/g1_29dof_walk.yaml` | Robot configuration file |
| `--policy` | `exported/policy_9999.pt` | Exported JIT policy that runs after pressing `A` |
| `--residual_policy` | `None` | *(Optional)* second policy that can be switched to at runtime with `R2` |
| `--encoder_seq_len` | `32` | History length of the student encoder (distillation policies) |
| `--enable_auto_switch` | off | Switch back to `--policy` automatically when the AprilTag is lost |
| `--max_detection_timeout` | `0.5` | Seconds without a tag detection before the automatic switch back |
| `--soft_limit_factor` | `1.7` | Scale of the joint range used by the joint-limit emergency stop (`1.0` = hardware limits) |

The policy type (standard / distillation) and the object observation size are detected automatically from the exported checkpoint. The RealSense camera is only opened for policies that use object observations.

## Startup Procedure

1. **Power on** the robot while it hangs from the gantry and wait until it enters *zero torque mode*.
2. **Debug mode:** press `L2 + R2` on the remote controller. The joints go into damping.
3. **Start the program** with the command above. The joints are in *zero torque* state.
4. **Default pose:** press `start`. The robot moves to the default joint pose over 2 s. Place the tray with the object on the tray holders and lower the gantry until the feet touch the ground.
5. **Run the policy:** press `A`. Use the joysticks to command the robot:
   - left stick forward / backward: `x` velocity
   - left stick left / right: `y` velocity
   - right stick left / right: yaw rate
6. **Policy switching** (only with `--residual_policy`):
   - `R2`: switch to the residual policy (only accepted while the AprilTag is visible)
   - `B`: switch back to the base policy
7. **Exit:** press `select` (or `Ctrl+C`). All motors enter damping mode and the program exits.

## Safety Mechanisms

- **Joint-limit emergency stop:** every control step, the target joint positions are checked against the joint limits in the config scaled by `--soft_limit_factor` (ankle pitch / roll are excluded). Any violation or non-finite target puts the robot into damping mode and stops the program.
- **Object visibility:** switching to the residual policy requires the AprilTag to be visible. With `--enable_auto_switch`, the controller falls back to the base policy when the tag has not been seen for `--max_detection_timeout` seconds.
- **Control-loop timing:** a warning is printed whenever one control step takes longer than `control_dt` (20 ms).

## Acknowledgements

The low-level control loop and remote-controller helpers are adapted from [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) (BSD 3-Clause, see [`LICENSE.unitree_rl_gym`](LICENSE.unitree_rl_gym)).
