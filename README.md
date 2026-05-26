# This the repo for reworking my  <a href="https://github.com/henryshum0/drone_fyp"> fyp </a> in Isaac Lab for Imitation Learning

This project runs under the Isaac Lab.   <a href="https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html"> Install Isaac Lab </a>


```bash
# clone
cd <your_isaaclab_root>/source/isaaclab_tasks/isaaclab_tasks/direct/
git clone https://github.com/henryshum0/drone_fyp
```

## Local Python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
touch ./.venv/COLCON_IGNORE
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```