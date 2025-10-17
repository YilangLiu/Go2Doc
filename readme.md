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

1. `cd unitree_sdk2_python`
2. `pip3 install -e .`

> [!NOTE]
> If encouter rrror when `pip3 install -e .`: ```bash
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH``` This error mentions that the cyclonedds path could not be found. Then try to following stpes:

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```
Enter the unitree_sdk2_python directory, set `CYCLONEDDS_HOME` to the path of the cyclonedds you just compiled, and then install unitree_sdk2_python.
```bash
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME="~/cyclonedds/install"
pip3 install -e .
```
For details, see: https://pypi.org/project/cyclonedds/#installing-with-pre-built-binaries

## Usage