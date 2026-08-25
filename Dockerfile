# GPU image for SeMoCo-Generator.
#
# Build (repo root):
#   docker build -t semoco-generator .
#
# The generator imports SeMoCo-Tokenizer in-process to decode motion codes back
# to geometry, so the tokenizer repo is cloned into the image and pointed at by
# $SOMA_TOKENIZER_ROOT. Both repos pin the same torch build for this reason.
#
# The kimodo submodule must be present at build time (clone with --recursive,
# or `git submodule update --init` first).
#
# Train (single node, 8 GPUs, code store mounted at /data):
#   docker run --rm --gpus all --ipc=host --shm-size=8g \
#     -v /path/to/semoco-MotionVerse:/data:ro -v /path/to/runs:/workspace/runs \
#     -e MOTIONVERSE_DATA_ROOT=/data \
#     semoco-generator \
#     torchrun --nproc_per_node=8 -m semoco_generator.train.train_t2m \
#       --config configs/t2m_150m_flan.yaml
#
# SMPL-X is registration-gated and cannot ship in the image. Mount it and set
# $SMPLX_MODEL_PATH when you need geometry decoding or the SMPL/HML track.

ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    SOMA_TOKENIZER_ROOT=/opt/SeMoCo-Tokenizer

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single-binary Python package manager).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# The frozen tokenizer: needed for decoding, and it carries the SOMA-X rig
# assets the SMPL/HML conversions read.
RUN git clone --recursive --depth 1 \
      https://github.com/OMEGA-i/SeMoCo-Tokenizer.git ${SOMA_TOKENIZER_ROOT} \
    && uv pip install --system --no-deps -e ${SOMA_TOKENIZER_ROOT}

# Full tree (see .dockerignore).
COPY . /workspace

# [decode] pulls the geometry stack, [tmr] the retrieval evaluator, [render]
# the MP4 writer. Training alone needs none of them, but the image is meant to
# cover train + generate + evaluate.
RUN uv pip install --system -e "/workspace[decode,tmr,render]"

# Default to a shell; pass the command after the image name, e.g.:
#   docker run --rm --gpus all semoco-generator \
#     python -m semoco_generator.eval.t2m_infer --checkpoint ... --text "a person walks"
CMD ["bash"]
