# ============================================================
# PyTorch 2.4 + CUDA 12.1 + Python 3.10 用 Dockerfile
# Pascal世代GPU (GTX 1080 Ti) 向け安定構成
# ============================================================

# NVIDIA公式ベースイメージ（CUDA 12.6.3 + Ubuntu 22.04）
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# システムアップデートとPython環境準備
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl wget vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# uvのインストール & パスを通す
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# libGLのインストール（OpenCV等で必要になる場合がある）
# 追加でOpenSlideツールもインストール
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx openslide-tools

# 作業ディレクトリ
WORKDIR /workspace

# bashを起動
CMD ["/bin/bash"]