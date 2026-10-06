#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE="$ROOT/vendor/FlashKDA"
if [[ -e "$SOURCE" ]]; then
  echo "Refusing to overwrite $SOURCE. Use a fresh checkout for setup." >&2
  exit 1
fi
mkdir -p "$ROOT/vendor"
git clone https://github.com/MoonshotAI/FlashKDA.git "$SOURCE"
git -C "$SOURCE" checkout 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b
git -C "$SOURCE" submodule update --init --depth 1 cutlass
python3 "$ROOT/scripts/patch_flashkda.py" "$SOURCE"
export FLASH_KDA_CUDA_ARCHS=${FLASH_KDA_CUDA_ARCHS:-100a}
export NVCC_THREADS=${NVCC_THREADS:-4}
export MAX_JOBS=${MAX_JOBS:-2}
export CC=${CC:-gcc}
export CXX=${CXX:-g++}
python3 -m pip install --no-build-isolation "$SOURCE"
