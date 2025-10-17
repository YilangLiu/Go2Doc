# Go2 Documentation 

This directory contains how to deploy Unitree Go2 in simulaiton and hardware using keyboard. The controller for simulation is trained wit custom environment in MuJoCo Playground.

## Installation


### Install Mujoco Playground
> [!IMPORTANT]
> Requires Python 3.10 or later.

1. `git clone git@github.com:google-deepmind/mujoco_playground.git && cd mujoco_playground`
2. Create a virtual environment: `conda create -n go2 python=3.11`
3. Activate it: `conda activate go2`
4. Install CUDA 12 jax: `pip install -U "jax[cuda12]"`
    * Verify GPU backend: `python -c "import jax; print(jax.default_backend())"` should print gpu
5. Install playground: `pip install -e ".[all]"`
6. `pip install onnxruntime hidapi`
7. Verify installation (and download Menagerie): `python -c "import mujoco_playground"`

### Install Unitree SDK2 Python

> [!NOTE]
> Do not pull unitree sdk2 python from offical unitree github.
