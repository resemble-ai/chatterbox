FROM ubuntu:20.04

# Install apt packages
RUN apt-get update && \
    apt-get install -y \
        wget curl git vim unzip \
        gcc g++ make \
    && \
    apt-get autoclean

# Set timezone
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -yq tzdata && \
    dpkg-reconfigure -f noninteractive tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get autoclean

# Install miniconda
RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Update in bashrc
RUN echo "source /opt/miniconda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda deactivate" >> /root/.bashrc

# Prepare conda env
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda config --add channels conda-forge && \
    conda tos accept && \
    conda create -n chatterbox python=3.10 -y && \
    conda activate chatterbox && \
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install toml-to-requirements && \
    pip cache purge && \
    conda clean --all

# COPY pyproject.toml and export requirements
# onnxruntime-gpu requires cudnn9, which is installed by pytorch 2.4.1,
# do not use an image with cudnn8
COPY pyproject.toml /opt/pyproject.toml
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate chatterbox && \
    cd /opt && \
    toml-to-req --toml-file pyproject.toml --optional-lists dev && \
    pip install -r requirements.txt && \
    pip cache purge

# COPY code
COPY . /workspace/chatterbox
# Install code
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate chatterbox && \
    cd /workspace/chatterbox && \
    pip install . && \
    pip cache purge

# Set working directory
WORKDIR /workspace/chatterbox

# Set entrypoint
ENTRYPOINT ["/bin/bash", "-c", "source /opt/miniconda/etc/profile.d/conda.sh && conda activate chatterbox && PYTHONPATH=/workspace/chatterbox python service/main.py --config_path service/config.py"]