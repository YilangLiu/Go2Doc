# Build : docker build -t mujoco_playground .
# Run : xhost +local:docker && docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$HOME/.Xauthority:/root/.Xauthority:rw" mujoco_playground bash
# Users can navigate to /app/mujoco_playground/learning to run scripts.
# Example : cd /app/mujoco_playground/learning &&  python train_jax_ppo.py --env_name=CartpoleBalance

# Use the specified NVIDIA JAX base image which includes Python, JAX, and CUDA.
FROM nvcr.io/nvidia/jax:25.01-py3

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies and git.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglew-dev \
    libosmesa6 \
    libglfw3 \
    patchelf \
    git && \
    # Clean up apt cache to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone the mujoco_playground repository into a subdirectory [2, 3, 4, 5].
RUN git clone https://github.com/google-deepmind/mujoco_playground.git ./mujoco_playground

# Copy the Go2Doc project files
COPY . /app/go2doc

# Upgrade pip and install the necessary Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mediapy wandb && \
    pip install --no-cache-dir -e /app/mujoco_playground && \
    pip install --no-cache-dir "jax[cuda12]" && \
    pip install --no-cache-dir warp onnxruntime hidapi

# Set the default command to launch a bash shell for interactive use.
ENV MUJOCO_GL=egl
CMD ["bash"]
