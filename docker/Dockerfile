FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu18.04

# install anaconda3
RUN apt-get update && apt-get install -y \
  curl \
  wget \
  git \
  vim \
  virtualenv

RUN curl -Lko /tmp/Anaconda3-2021.11-Linux-x86_64.sh https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh && \
	chmod +x /tmp/Anaconda3-2021.11-Linux-x86_64.sh && \
	bash /tmp/Anaconda3-2021.11-Linux-x86_64.sh -b -p /root/anaconda && \
	rm -rf /tmp/Anaconda3-2021.11-Linux-x86_64.sh

ENV PATH="/root/anaconda/bin:$PATH"

# install mujoco: https://github.com/openai/mujoco-py/blob/master/Dockerfile
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

#    && wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz -O mujoco.tar.gz \
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV MUJOCO_GL "egl"
#ENV MUJOCO_PY_MUJOCO_PATH /root/.mujoco/mujoco-2.1.1


# Copy over just mujoco requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
WORKDIR /tmp/dep
COPY ./requirements.mujoco.txt /tmp/dep
COPY ./requirements.mujoco.dev.txt /tmp/dep

RUN pip install --no-cache-dir -r requirements.mujoco.txt
RUN pip install --no-cache-dir -r requirements.mujoco.dev.txt
RUN pip install "mujoco-py<2.2,>=2.1"
RUN python -c "import mujoco_py"

# enable mujoco headless gpu rendering
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
RUN mkdir -p /usr/lib/nvidia-000

# install dm_control compatible with mujoco210
RUN pip install dm-control==0.0.403778684

# project specific requirements
## d4rl had to be installed from a modified source (because of setup.py which forces dm_control to get updated to the latest version)
COPY d4rl /tmp/dep/d4rl
RUN cd d4rl && pip install -e . && cd ..

COPY ./requirements.txt /tmp/dep
RUN pip install -r requirements.txt

