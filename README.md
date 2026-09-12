# Installation

Clone the repo:

```git clone https://github.com/LampLighterLab/BigRedGym.git```

Then, enter the BigRedGym folder.

Create the venv using uv:

```uv sync --frozen```

Note: you will need to use the ```--frozen``` flag every time you run a python script in this repo when the optional VSim wheel or Unitree SDK checkout is absent. example: ```uv run --frozen scripts/train.py --task=go2trot  --device=cuda:0 --headless --max_iterations=300```

# Training

To train, run ```scripts/train.py```:

```
uv run --frozen scripts/train.py --task=go2trot
```

Mandatory arguments:
```
--task: the name of your environment (go2, go2trot, mini_cheetah, etc)
```

Optional arguments:
```
--device: cpu or cuda:0 (nvidia gpu)
--backend: mujoco or vsim (most likely you will only use mujoco)
--num_envs: how many environments to run in parallel in training
--max_iterations: how many training iterations to run (typically 300-500 is sufficient for the go2)
--headless: disable the viewer window during training
--save_interval: override checkpoint interval in learning iterations
--seed: random seed for rng
--batch_size: override batch size from cfg
--experiment_name: Override experiment_name (log dir is logs/<experiment_name>/...)
--resume: Resume optimizer and model state from an existing run
--load_run: Run directory under logs/<experiment_name>/ (default: latest).
--checkpoint: Checkpoint iteration to resume (default: latest).
--original_cfg: Load environment and runner configs saved with the selected run.
```

# Testing

To test your policy in simulation, run ```scripts/play.py```:

```
uv run --frozen scripts/play.py --task=go2trot
```

Mandatory arguments:
```
--task: the name of your environment (go2, go2trot, mini_cheetah, etc)
```

Optional arguments:
```
--device: cpu or cuda:0 (nvidia gpu)
--backend: mujoco or vsim (most likely you will only use mujoco)
--experiment_name: Override experiment_name (log dir is logs/<experiment_name>/...)
--load_run: Run directory under logs/<experiment_name>/ (default: latest).
--checkpoint: Checkpoint iteration to resume (default: latest).
--original_cfg: Load environment and runner configs saved with the selected run.
```

The automated regression suite runs with `uv run --frozen python -m pytest -q`.
See [limitations and validation scope](genAI_skills/README_MUJOCO.md#limitations-and-validation-scope)
for what the retained tests cover.

# Note on AI-generated files
Some of the code has been worked on by an AI agent, in the case where code segments have been heavily edited by AI, the AI-generated files in [genAI_skills](genAI_skills/) may be of some value in understanding the code, especially [README_MUJOCO.md](genAI_skills/README_MUJOCO.md).
