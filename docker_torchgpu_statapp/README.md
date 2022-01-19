# Getting started with docker on Linux

Why Docker?

1. It contains all related things for the AI tasks.
2. Make sure all members in team working on the same environement (the same version of libs, the same ports, the same folder,...).
3. You don't have to install (by yourself) all the necessary things (and it always makes unexpected errors). For example, installing tensorflow with GPU on Linux is a very bad experience. With docker, you only need to install sucessfully GPU driver on your lap, the rest will be done by docker.

## Install docker

Follow [the official guide](https://docs.docker.com/engine/install/). Normally, below steps

``` bash
# update system
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker’s official GPG key:
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# install "stable" release
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# install docker engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

# check if ok?
sudo docker run hello-world
# it will print something to say it works

# incase docker-compose isn't installed
sudo apt-get install docker-compose

# run docker with root? (no need to use "sudo" each time)
sudo groupadd docker # create a docker group
sudo usermod -aG docker <user> # <user> is your username on your computer
newgrp docker # activate the changes

# start docker on boot
sudo systemctl enable docker
```

## Make docker work with GPU

In case your computer has NVIDIA GPU, you can follow below steps to make docker work with GPU.

**Make sure that you have already installed NIVIDIA driver sucessfully on your machine.**

``` bash
# verify that your computer has a graphic card
lspci -nn | grep '\[03'

# run to check
nvidia-smi
# output: NVIDIA-SMI 450.80.02 Driver Version: 450.80.02    CUDA Version: 11.0
# it's maximum CUDA version that your driver supports
```

``` bash
# install
sudo apt-get install -y nvidia-cuda-toolkit

# then check
# check current version of cuda
nvcc --version
# If there is not nvcc, it may be in /usr/local/cuda/bin/
# Add this location to PATH
# modify ~/.zshrc or ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
```

Follow [this official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) to install `nvidia-docker2` (ignore the first step which talks about the installing docker, you've done that). Below are the steps,

``` bash
# add "stable" version
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# then install
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# restart docker
sudo systemctl restart docker

# check
dpkg -l | grep nvidia-docker
```

``` bash
# Verifying –gpus option under docker run
docker run --help | grep -i gpus
# output: --gpus gpu-request GPU devices to add to the container ('all' to pass all GPUs)
```

Below is the final step to be sure that your GPU is linked to your docker.

``` bash
# Listing out GPU devices
docker run -it --rm --gpus all ubuntu nvidia-smi -L

# It should return all gpu devices on your computer, something like
# output: GPU 0: GeForce GTX 1650 (...)
```

## Setting up container

Go into the directory, where the 'docker-compose.yml' file is and run :

``` bash
# build an image and create the container at the same time
docker-compose up
```

``` python
# check if tensorflow + gpu works
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

In the future, each time you start your computer

``` bash
# Check if docker is running
docker ps
# if something like below, it's running
# CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES

# Run the container
docker start container_tfgpu # or container_nvidiagpu

# If you wanna check currently running containers
docker ps

# Stop container
docker stop container_tfgpu # or container_nvidiagpu

# Enter the running container (to install packages, for example)
docker exec -it container_tfgpu bash # or container_nvidiagpu bash

# Inside this, if you wanna install some package, use pip
pip install <package_name>
```
