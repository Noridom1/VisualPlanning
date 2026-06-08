#!/bin/bash

set -euo pipefail

TASK="${1:-all}"

download_model() {
  local task="$1"
  local repo="$2"
  local dest="$3"

  echo ">>> Downloading $task model from $repo"
  huggingface-cli download "$repo" --local-dir "$dest"
}

case "$TASK" in
  frozenlake)
    download_model frozenlake yixu1/VPRL-7B-FrozenLake ./models/frozenlake/VPRL_merged_ckpts
    ;;
  maze)
    download_model maze yixu1/VPRL-7B-Maze ./models/maze/VPRL_merged_ckpts
    ;;
  minibehaviour)
    download_model minibehaviour yixu1/VPRL-7B-MiniBehaviour ./models/minibehaviour/VPRL_merged_ckpts
    ;;
  all)
    download_model frozenlake yixu1/VPRL-7B-FrozenLake ./models/frozenlake/VPRL_merged_ckpts
    download_model maze yixu1/VPRL-7B-Maze ./models/maze/VPRL_merged_ckpts
    download_model minibehaviour yixu1/VPRL-7B-MiniBehaviour ./models/minibehaviour/VPRL_merged_ckpts
    ;;
  *)
    echo "Usage: bash scripts/download_models.sh [frozenlake|maze|minibehaviour|all]"
    echo "Defaults to 'all' if no argument is given."
    exit 1
    ;;
esac

echo ">>> Download complete."
