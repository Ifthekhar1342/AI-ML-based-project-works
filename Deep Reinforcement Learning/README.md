
# Physics-Informed RL Toolpath Planning for Powder Bed Fusion

This project is a runnable reference implementation of the methodology described in:

**Learning-Based Toolpath Planning for Bed-Based Additive Manufacturing**

It reproduces the methodological structure rather than the authors' exact source code.

## Included

- Procedural target-mask generation
  - Training: letters
  - Validation: symbols
  - Testing: held-out geometric shapes
- Geometric toolpath environment
- Thermal environment coupled to a structured Q4 finite-element heat solver
- Five actions: up, down, left, right, and laser on/off toggle
- Composite reward:
  - fill target
  - penalize spill
  - penalize revisits
  - penalize each time step
- DQN and PPO training through Stable-Baselines3
- Zigzag baseline
- Evaluation utilities and visualization
- Jupyter notebook walkthrough

## Scope and limitations

The implementation follows the manuscript's controlled benchmark assumptions:

- 2-D
- single layer
- structured 10×10 or 20×20 grids
- fixed source power and width
- simplified transient conduction
- no latent heat, melt-pool fluid dynamics, recoil pressure, vaporization, or temperature-dependent material properties

It is intended for research demonstration and extension, not direct machine control.

## Installation

```bash
python -m pip install -r requirements.txt
```

For a GPU-enabled PyTorch installation, install the appropriate PyTorch build for your CUDA version first, then install the remaining packages.

## Quick smoke tests

The core environments and FE solver work without Stable-Baselines3:

```bash
python pbf_rl_toolpath.py smoke --env geometric --steps 20
python pbf_rl_toolpath.py smoke --env thermal --steps 20
python pbf_rl_toolpath.py zigzag --env thermal
```

## Train an agent

A short demonstration run:

```bash
python pbf_rl_toolpath.py train \
  --algo ppo \
  --env thermal \
  --steps 100000 \
  --output ppo_thermal
```

DQN example:

```bash
python pbf_rl_toolpath.py train \
  --algo dqn \
  --env geometric \
  --steps 100000 \
  --output dqn_geometric
```

The manuscript-scale study uses substantially more simulation experience than these demonstration defaults.

## Python use

```python
from pbf_rl_toolpath import ThermalPBFEnv, plot_episode

env = ThermalPBFEnv(n=10, split="test", seed=0, fixed_mask_name="ring")
obs, info = env.reset()

for _ in range(40):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

plot_episode(env)
```

## Main extension points

1. Add variable laser power and scan speed to the action space.
2. Add explicit thermal objectives:
   - peak-temperature penalty
   - temperature-gradient penalty
   - cooling-rate penalty
3. Replace the simplified material model with temperature-dependent properties and latent heat.
4. Extend from 2-D single-layer masks to 3-D multi-layer toolpaths.
5. Add sensor observations and sim-to-real calibration.
6. Replace the FE solver with a surrogate or multi-fidelity model for faster training.
