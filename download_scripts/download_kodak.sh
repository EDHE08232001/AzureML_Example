#!/bin/zsh

# Create the directory if it doesn't exist
mkdir -p datasets/kodak

# Loop through and download
for i in $(seq -w 1 24); do
  curl -L "https://r0k.us/graphics/kodak/kodak/kodim${i}.png" \
       -o "datasets/kodak/kodim${i}.png"
done