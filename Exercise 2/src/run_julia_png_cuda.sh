#!/bin/bash
set -e

# Compile the CUDA code
nvcc -o julia_png_cuda julia_png_cuda.cu ./lodepng/lodepng.cpp

# Parameter sets to try
RE_VALUES=(-0.8 -0.9 -0.70176 -0.4 -0.835 -0.123 -0.70176)
IM_VALUES=(0.156 0.4 0.3842 0.6 -0.2321 0.745 -0.3842)
ITER_VALUES=(16 32 64 128)
X_DOMAINS=("-1 1" "-2.0 2.0" "-5.0 5.0")
Y_DOMAINS=("-1 1" "-2.0 2.0" "-5.0 5.0")
TRESHOLDS=(2.5 5 10 20 40 80)
COLOR_MAPS=(0)

# Run combinations
for i in "${!RE_VALUES[@]}"; do
  re="${RE_VALUES[$i]}"
  im="${IM_VALUES[$i]}"
  for iter in "${ITER_VALUES[@]}"; do
    for j in "${!X_DOMAINS[@]}"; do
      for th in "${TRESHOLDS[@]}"; do
        
        # Get the corresponding x and y domains
        xdomain="${X_DOMAINS[$j]}"
        ydomain="${Y_DOMAINS[$j]}"
        
        x_min=$(echo $xdomain | cut -d' ' -f1)
        x_max=$(echo $xdomain | cut -d' ' -f2)
        y_min=$(echo $ydomain | cut -d' ' -f1)
        y_max=$(echo $ydomain | cut -d' ' -f2)

        echo "Running: c=($re,$im), iter=$iter, x=[$x_min,$x_max], y=[$y_min,$y_max], cmap=$cmap", "thresh=$th"
        for cmap in "${COLOR_MAPS[@]}"; do
          ./julia_png_cuda "$re" "$im" "$iter" "$x_min" "$x_max" "$y_min" "$y_max" "$cmap" "$th"
        done
      done
    done
  done
done
