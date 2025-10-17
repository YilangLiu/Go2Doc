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
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH``` This error mentions that the cyclonedds path could not be found. Then try the following steps:

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

### Simulation Evaluation
```bash
python play_go2_keyboard.py
```

### Hardware Testing
> [!Warning]
> Always test your controller in the simulation before deploying on real robots

> [!Caution]
> Wait until the robot is fully started (standing pose) to run the deployment scripts. 

#### Configure the networks. 
1. Connect one end of the network cable to the Go2 robot, and the other end to the user's computer. Turn on the USB Ethernet of the computer and configure it. The IP address of the onboard computer of the machine dog is 192.168.123.161, so it is necessary to set the USB Ethernet address of the computer to the same network segment as the machine dog. For example, entering 192.168.123.222 ("222" can be changed to other) in the Address field.
<img src="media/image1.png" width="400">
<img src="media/image2.png" width="400">
To test whether the user's computer is properly connected to the built-in computer of the Go2 robot, you can enter ping 192.168.123.161 in the terminal for testing. The connection is successful when something similar to the following appears.
<img src="media/image3.png" width="400">

2. View the network card names of the 123 network segment through the ifconfig command, as shown in the following figure:
<img src="media/image4.png" width="400">
As shown in the above figure, the network card name corresponding to the IP address 192.168.123.222 is enxf8e43b808e06. Users need to remember this name as it will be a necessary parameter when running the routine.

3. Run the deployment example 
```bash
python traj_following_example.py enxf8e43b808e06
```