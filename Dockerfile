# ============================================================
# PyTorch 2.4 + CUDA 12.6 + Python 3.10 用 Dockerfile
# Pascal世代GPU (GTX 1080 Ti) 向け安定構成
# ============================================================

# NVIDIA公式ベースイメージ
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# ------------------------------------------------------------
# システムアップデート + 必須パッケージ + OpenSlide + libvips
# すべてまとめてレイヤー削減
# ------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl wget vim build-essential \
    libglib2.0-0 libgl1-mesa-glx \
    openslide-tools libvips-tools \
    openjdk-17-jdk-headless \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:${PATH}"

# ------------------------------------------------------------
# uv のインストール
# ------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ------------------------------------------------------------
# 作業ディレクトリ
# ------------------------------------------------------------
WORKDIR /workspace

CMD ["/bin/bash"]