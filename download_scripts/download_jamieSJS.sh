#!/bin/zsh
python3 download_scripts/download_imagenet.py \
    --dataset JamieSJS/imagenet-10 \
    --num-images 13000 \
    --split test \
    --out-dir datasets/imagenet/train