<div align="center">

# SteadyTray

### Learning Object Balancing Tasks in Humanoid Tray Transport via Residual Reinforcement Learning

**UC San Diego**

[[Project Page]](https://steadytray.github.io/) [[Paper]](https://arxiv.org/abs/2603.10306) [[Video]](https://youtu.be/hBYnM1GcxbU)

</div>

This is the official code release for **SteadyTray**. The repository contains the training pipeline and sim2sim deployment code.

## Overview

This project provides reinforcement learning environments for training steady-tray tasks on Unitree robots, built on top of a custom fork of [IsaacLab](https://github.com/AllenHuangGit/IsaacLab). The training pipeline uses PPO via RSL-RL and supports multi-GPU distributed training. Trained policies can be deployed in MuJoCo for sim2sim validation.

## Installation

This project depends on a custom fork of IsaacLab. The recommended setup is via Docker.

### 1. Clone the Repositories

```bash
# Clone our IsaacLab fork
git clone https://github.com/AllenHuangGit/IsaacLab.git

# Clone SteadyTray
git clone https://github.com/AllenHuangGit/steadytray.git
```

### 2. Build and Launch the Isaac Lab Docker Container

Follow the [Isaac Lab Docker guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html) to build the Docker image from our IsaacLab fork, then launch the container with the SteadyTray repository mounted.

**Option A:** Uncomment the mount lines in [`docker/docker-compose.yaml` (L68–L71)](https://github.com/AllenHuangGit/IsaacLab/blob/main/docker/docker-compose.yaml#L68-L71) and update the source path to your local SteadyTray directory:

```yaml
    # * The following is used to mount our extension codebase
  - type: bind
    source: /path/to/steadytray     #* Change this to the path of your local extension codebase
    target: /workspace/steadytray
```

Then use the standard IsaacLab Docker commands:

```bash
cd IsaacLab/docker
./container.py start
```

**Option B:** Mount the volume directly via the command line:

```bash
cd IsaacLab/docker
# Build the Docker image
docker compose --env-file .env.base --profile base build isaac-lab-base
# Launch the container with SteadyTray mounted
docker compose --env-file .env.base --profile base run --rm \
    -v /path/to/steadytray:/workspace/steadytray \
    isaac-lab-base bash
```

### 4. Install SteadyTray

Inside the container, install the package in editable mode:

```bash
cd /workspace/steadytray
python -m pip install -e source/steadytray

# Install git-lfs and tmux
apt-get update && apt-get install -y git-lfs
apt update && apt install -y tmux
```

## Training

The training pipeline consists of four sequential stages (see Appendix B of the [paper](https://arxiv.org/abs/2603.10306) for details). Stages 2–4 each require loading the checkpoint from the previous stage.

| Stage | Task | Description | Requires Pretrained Model |
|---|---|---|---|
| 1 | `G1-Steady-Tray-Pre-Locomotion` | Base locomotion with upper body frozen for faster training | No |
| 1* | `G1-Steady-Tray-Locomotion` | *(Optional)* Full-body locomotion training | No |
| 2 | `G1-Steady-Tray` | Fine-tune locomotion with tray-holding rewards | Yes (Stage 1) |
| 3 | `G1-Steady-Object` | Residual teacher for object stabilization on the tray | Yes (Stage 2) |
| 4 | `G1-Steady-Object-Distillation` | Distill privileged teacher into a deployable student policy | Yes (Stage 3) |

### Stage 1: Pre-train Locomotion

Train the base whole-body locomotion policy (upper body frozen):

```bash
python scripts/rsl_rl/train.py \
    --task G1-Steady-Tray-Pre-Locomotion \
    --num_envs 4096 \
    --headless \
    --run_name "pretrain_loco" \
    --max_iterations 10000
```

### Stage 2: Tray-Holding Fine-tune

Fine-tune the locomotion policy with tray-specific rewards. Load the Stage 1 checkpoint:

```bash
python scripts/rsl_rl/train.py \
    --task G1-Steady-Tray \
    --num_envs 4096 \
    --headless \
    --resume \
    --load_run <stage1_run_dir> \
    --run_name "tray_finetune" \
    --max_iterations 10000
```

### Stage 3: Residual Object Stabilization (Teacher)

Train the residual module to stabilize objects on the tray. Load the Stage 2 checkpoint:

```bash
python scripts/rsl_rl/train.py \
    --task G1-Steady-Object \
    --num_envs 4096 \
    --headless \
    --resume \
    --load_run <stage2_run_dir> \
    --run_name "residual_teacher" \
    --max_iterations 10000
```

### Stage 4: Distillation (Student)

Distill the privileged teacher into a deployable student policy. Load the Stage 3 checkpoint:

```bash
python scripts/rsl_rl/train.py \
    --task G1-Steady-Object-Distillation \
    --num_envs 4096 \
    --headless \
    --resume \
    --load_run <stage3_run_dir> \
    --run_name "distillation" \
    --max_iterations 10000
```

### Multi-GPU Distributed Training

Any stage can be run with multi-GPU distributed training:

```bash
python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=2 \
    scripts/rsl_rl/train.py \
    --task <TASK_NAME> \
    --num_envs 4096 \
    --headless \
    --distributed \
    --run_name "my_run" \
    --max_iterations 10000
```

### Inference

A pretrained student model is provided in `model/model_9999.pt`. Visualize it in IsaacSim:

```bash
python scripts/rsl_rl/play.py \
    --task G1-Steady-Object-Distillation \
    --checkpoint "model/model_9999.pt"
```

You can also run inference with your own trained checkpoints:

```bash
python scripts/rsl_rl/play.py \
    --task <TASK_NAME> \
    --checkpoint "logs/rsl_rl/<task_dir>/<run_name>/model_<iter>.pt"
```

## Sim2Sim Deployment (MuJoCo)

A lightweight MuJoCo deployment is provided for visualizing trained policies without requiring Isaac Sim. See [deploy/deploy_mujoco/README.md](deploy/deploy_mujoco/README.md) for full instructions.

Quick start:

```bash
cd deploy/deploy_mujoco
conda env create -f environment.yml
conda activate g1_deploy

python deploy_mujoco.py --policy exported/policy_9999.pt --config deploy/configs/g1_29dof_walk.yaml
```

## Acknowledgements

This codebase is built upon the following awesome open-source projects:

- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab): RL training framework for Unitree robots on IsaacLab.
- [g1_deploy_mujoco](https://github.com/RoboCubPilot/g1_deploy_mujoco): Lightweight MuJoCo deployment for G1 policies.
- [IsaacLab](https://github.com/isaac-sim/IsaacLab): Foundation for training and simulation.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{huang2026steadytraylearningobjectbalancing,
      title={SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport via Residual Reinforcement Learning}, 
      author={Anlun Huang and Zhenyu Wu and Soofiyan Atar and Yuheng Zhi and Michael Yip},
      year={2026},
      eprint={2603.10306},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.10306}, 
}
```
