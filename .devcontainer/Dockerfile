#adapted from https://github.com/JuliaGPU/julia-ngc
# ARG IMAGE=nvidia/cuda:12.1.1-devel-ubuntu20.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# julia
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install --yes --no-install-recommends \
                    # basic stuff
                    curl \
    git \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    xvfb \
    scrot \
    x11-utils \
    xdotool \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt install -y python3 && \
    apt-get install -y python3-dev python3-distutils
    
RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN apt-get update -q -y && \
        DEBIAN_FRONTEND=noninteractive \
        apt-get install -q -y --allow-change-held-packages\
        vim net-tools curl \
        libgl1-mesa-glx \
        xserver-xorg-video-dummy \
        libxrender1 \
        libpulse0 \
        libpulse-mainloop-glib0  \
        libnss3  \
        libxcomposite1 \
        libxcursor1 \
        libfontconfig1 \
        libxrandr2 \
        libasound2 \
        libglu1 \
        x11vnc \
        awesome \
        jq \
        nautilus\
        jupyter-core\
        zip\
        p7zip-full\
        apt-utils\
        octave\
        kmod\
        zlib1g\
        python3-dev\
        bzip2\
        cmake\
        libfreetype6-dev \
        curl\
        libzmq3-dev \
        software-properties-common\
        libhdf5-serial-dev\
        git \
        at-spi2-core \
        libgtk-3-dev \
        xauth \
        xvfb \
        ninja-build \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        ninja-build \
        wget \
        doxygen \
        autoconf \
        automake \
        cmake \
        g++ \
        gcc \
        make \
        nasm \
        xxd \
        yasm \
        libglfw3 \
        libglfw3-dev \
        freeglut3-dev \
                make \
        pkgconf \
        xz-utils \
        xorg-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        libxxf86vm-dev \
        libglfw3 libglfw3-dev xorg-dev libxkbcommon-dev meson clang git cmake libssl-dev cmake \
        ffmpeg && \
        apt-get install -q -y --reinstall ca-certificates

RUN python3 -m pip install --no-cache-dir \
    gdown==3.3.1 monai==1.3.0 nibabel torch

RUN apt-get install --yes --no-install-recommends wget build-essential libcurl4 && \
    wget https://curl.se/download/curl-7.81.0.tar.gz && \
    tar -xvf curl-7.81.0.tar.gz && cd curl-7.81.0 && \
    ./configure --with-openssl && make && make install

COPY download_data.py /workspace/download_data.py
RUN python3 /workspace/download_data.py

RUN python3 -m pip install --no-cache-dir SimpleITK

ARG JULIA_RELEASE=1.12
ARG JULIA_VERSION=1.12.5
RUN curl -s -L https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_RELEASE}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz | \
    tar -C /usr/local -x -z --strip-components=1 -f -

COPY Project.toml LocalPreferences.toml /usr/local/share/julia/environments/v${JULIA_RELEASE}/

RUN JULIA_DEPOT_PATH=/usr/local/share/julia \ 
    julia -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

# generate the device runtime library for all known and supported devices
# XXX: this doesn't precompile into the system depot anymore
RUN JULIA_DEPOT_PATH=/usr/local/share/julia \ 
    julia -e 'using CUDA; CUDA.precompile_runtime()'

# the system depot contains precompiled packages, but its global environment cannot be
# active at the same time of the user environment. we solve this by using a startup script
# that will initialize the user depot with the Project and Manifest from the system depot.
# however, for that script to run, we need to start with only the system depot active.
ENV JULIA_DEPOT_PATH=/usr/local/share/julia
COPY startup.jl /usr/local/share/julia/config/

# user environment
RUN mkdir -m 0777 /data
RUN mkdir $HOME/data

ENV JULIA_HISTORY=/data/logs/repl_history.jl
ENV LD_LIBRARY_PATH=""

# setup enviromental variables for Julia multithreading keeping one interactive thread
COPY setup_threads.sh  /etc/profile.d/

WORKDIR "/workspace"
COPY examples /workspace/examples

ENTRYPOINT ["/usr/local/bin/julia"]
CMD ["sleep", "infinity"]
