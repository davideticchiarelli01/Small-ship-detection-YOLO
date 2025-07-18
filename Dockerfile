# Utilizza CUDA 11.8 come immagine di base
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

LABEL authors="Davide Ticchiarelli, Giampaolo Marino, Niccolò Ciotti"

# Aggiornamento dei pacchetti e installazione delle dipendenze di sistema
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    unzip \
    wget \
    tmux \
    nano \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libgl1-mesa-glx \
    python3-pip \
    sudo \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Imposta Python 3 come predefinito
RUN ln -s /usr/bin/python3 /usr/bin/python

# Aggiorna pip e installa le librerie Python necessarie
RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools packaging

# Installazione di PyTorch, Torchvision e Torchaudio compatibili con CUDA 11.8
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Installazione delle altre dipendenze Python
RUN python -m pip install \
    wandb \
    python-dotenv \
    codecarbon \
    scikit-learn \
    matplotlib \
    opencv-python-headless \
    ttach \
    ultralytics \
    pyyaml \
    tqdm

# Imposta la directory di lavoro
WORKDIR /app

# Imposta il comando di avvio per aprire una shell bash
CMD ["/bin/bash"]
