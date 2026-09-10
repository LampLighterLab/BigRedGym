# Installation

Clone the repo:

```git clone https://github.com/LampLighterLab/BigRedGym.git```

Then, enter the BigRedGym folder.

Create the venv using uv:

```uv sync --frozen```

Note: you will need to use the ```---frozen``` flag every time you run a python script in this repo if you do not install unitree-sdk2py (only used for deploying on the hardware). example: ```uv run --frozen scripts/train.py --task=go2trot  --device=cuda:0 --headless --max_iterations=300```