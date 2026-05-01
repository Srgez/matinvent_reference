#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
CACHE_DIR="${ROOT_DIR}/.uv-cache"
MPL_CONFIG_DIR="${ROOT_DIR}/.mplconfig"

cd "${ROOT_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${MPL_CONFIG_DIR}"
export UV_CACHE_DIR="${CACHE_DIR}"
export MPLCONFIGDIR="${MPL_CONFIG_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

uv venv "${VENV_DIR}" --python 3.10 --allow-existing
source "${VENV_DIR}/bin/activate"

# The project relies on CUDA-specific PyTorch wheels and extension wheels
# published across multiple indexes, so we keep the documented requirements
# file as the single source of truth for uv-based installation.
uv pip install --requirements requirements.txt --index-strategy unsafe-best-match
