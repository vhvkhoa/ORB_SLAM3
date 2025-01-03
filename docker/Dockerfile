FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific
SHELL [ "/bin/bash", "-c" ]
ENV HOME=/workspace

RUN apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    build-essential \
    cmake \
    curl \
    g++ \
    wget \
    bzip2 \
    git \
    vim \
    tmux \
    htop \
    zip \
    unzip \
    ca-certificates \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    libglu1-mesa \
    libxext6 \
    libxtst6 \
    libxrender1 \
    libxi6 \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    libyaml-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libavutil-dev \
    libavfilter-dev \
    libavdevice-dev \
    libswresample-dev \
    less \
    groff \
    mpich \
    iputils-ping \
    kbd \
    libgtk2.0-dev \
    pkg-config \
    python3-pip

# Add ORB_SLAM3 dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
	lsb-release \
    libgl1-mesa-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    libglew-dev \
    libboost-all-dev \
    libssl-dev \
    libeigen3-dev \
    libepoxy-dev

# Clone and build OpenCV
WORKDIR /workspace/Dev
RUN git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    git checkout 4.10.0 && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && \
    make install

# Clone and build Pangolin
WORKDIR /workspace/Dev
RUN git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && cd build && \
    cmake .. -D CMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

# RealSense dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    udev \
    libssl-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libcanberra-gtk-module
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | \
    tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev

WORKDIR /workspace
