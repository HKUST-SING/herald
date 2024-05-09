FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
ENV CUDA_VERSION=11.3.1
ENV CUDNN_VERSION=8.2.0.53
ARG REGION

ARG http_proxy
ARG https_proxy

ENV USE_CUDA_PATH /usr/local/cuda:/usr/local/cudnn/lib64
ENV PATH /usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nccl/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/cudnn/lib64:/usr/local/cuda/lib64:$LIBRARY_PATH

ARG CUDNN_VERSION=$CUDNN_VERSION-1+cuda$CUDA_VERSION

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update -qq
RUN apt-get install -y software-properties-common
RUN apt-get install -y --allow-unauthenticated --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        tzdata \
        ca-certificates \
        git \
        curl \
        wget \
        vim \
        rsync \
        sudo \
        bison \
        iproute2 \
        iputils-ping \
        iftop \
        unzip \
        perftest \
        lsb-release \
        numactl \
        libssl-dev \
        libnuma-dev \
        ibverbs-providers \
        librdmacm-dev \
        ibverbs-utils \
        rdmacm-utils \
        libibverbs-dev \
        sshpass \
        g++-4.8 \
        gcc \
        tmux \
        openssh-server \
        openssh-client


# install python3.8
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.8 python3.8-dev python3-pip
RUN python3.8 -m pip install pip
RUN python3.8 -m pip install pip --upgrade
RUN python3.8 -m pip install --no-cache-dir setuptools

# Install NCCL
ENV NCCL_VERSION=2.8

RUN apt purge libnccl2 -y --allow-unauthenticated --allow-downgrades --allow-change-held-packages --no-install-recommends &#& \
    cd / && \
    wget -q -O - https://github.com/NVIDIA/nccl/archive/v$NCCL_VERSION.tar.gz | tar -xzf - && \
    cd nccl-$NCCL_VERSION && make -j src.build && make pkg.txz.build && \
    mkdir -p /usr/local/nccl && \
    tar -Jxf /nccl-$NCCL_VERSION/build/pkg/txz/nccl*.txz -C /usr/local/nccl/ --strip-components 1 && \
    echo "/usr/local/nccl/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig && rm -rf /nccl-$NCCL_VERSION


WORKDIR /root/

# install cmake
RUN cd /root && wget -q -O - https://github.com/Kitware/CMake/releases/download/v3.18.0/cmake-3.18.0.tar.gz | tar -xzf - && \
    cd cmake-3.18.0 && ./bootstrap && make -j8 && make install

# install mpi
RUN cd /root && wget -q -O - https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz | tar -xzf - && \
    cd openmpi-4.0.3 && ./configure && make -j8 && make install

ENV MPI_HOME=/root/openmpi-4.0.3

# install mkl-dnn
RUN cd /root && wget -q -O - https://github.com/intel/mkl-dnn/archive/v1.6.1.tar.gz | tar -xzf - && \
    cd oneDNN-1.6.1 && mkdir -p build && cd build && cmake .. && make -j8 && make install
ENV MKL_ROOT=/root/oneDNN-1.6.1
ENV MKL_BUILD=/root/oneDNN-1.6.1/build

# install zeromp
RUN cd /root && wget -q https://github.com/zeromq/libzmq/releases/download/v4.3.2/zeromq-4.3.2.tar.gz && \
    tar -xzf zeromq-4.3.2.tar.gz && \
    cd zeromq-4.3.2 && mkdir -p build && cd build && cmake .. && make -j8 && make install
ENV ZMQ_ROOT=/root/zeromp-4.3.2


# install protobuf
RUN cd /root && wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.15.8/protobuf-all-3.15.8.tar.gz && \
    tar -xzf protobuf-all-3.15.8.tar.gz && \
    cd protobuf-3.15.8 && ./configure && make -j8 && make install

# install pybind11
RUN cd /root && wget -q https://github.com/pybind/pybind11/archive/refs/tags/v2.6.0.tar.gz && \
    tar -xzf v2.6.0.tar.gz && \
    cd pybind11-2.6.0 && \
    mkdir build && cd build && cmake .. && make -j8 && make install


RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cudnn/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig

# GOLANG PATH
ENV PATH /usr/local/go/bin:${PATH}

################################ install your framework ################################
# install DNN frameworks
ARG TENSORFLOW_VERSION=2.2.0
ARG MXNET_VERSION=1.9.1
RUN python3 -m pip install --upgrade pip
RUN python3.8 -m pip --no-cache-dir install \
    tensorflow-gpu==$TENSORFLOW_VERSION

RUN python3.8 -m pip --no-cache-dir install torch torchvision
RUN python3.8 -m pip --no-cache-dir install pybind11==2.6.0 metis
RUN python3.8 -m pip install pyyaml paramiko psutil boto3 tqdm scikit-learn pandas

# we may need this due to some bugs
RUN python3.8 -m pip install --upgrade scipy
RUN python3.8 -m pip install scikit-learn

RUN rm -rf /usr/local/cuda/lib64/libcuda.so && \
    rm -rf /usr/local/cuda/lib64/libcuda.so.1

# install clang-14 for c++-20
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
RUN add-apt-repository -y "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-14 main"
RUN apt update
RUN apt install clang-14 -y
  
# make 3.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2

# install boost 
RUN apt install -y libboost-all-dev

# install gtest
RUN apt install -y libgtest-dev
RUN cd /usr/src/gtest && mkdir -p build && cd build && cmake .. && make && make install
