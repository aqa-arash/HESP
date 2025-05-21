#!/bin/bash

# filepath: c:\Users\Christian Karg\Documents\FAU_C\HESP\ex1\src\stream\run_benchmark.sh

# Ausführbare Dateien
BASE_EXEC="../../build/stream/stream-base"
OMP_EXEC="../../build/stream/stream-omp-host"
CUDA_EXEC="../../build/stream/stream-cuda"

# Puffergrößen (nx) in einer Liste definieren
BUFFER_SIZES=(1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456)

# Ausgabe-Datei
OUTPUT_FILE="benchmark_results.txt"

# Überschreibe die Datei mit einer Kopfzeile
echo "Buffer Size, Version, Elapsed Time (ms), MLUPps, Bandwidth (GBps)" > $OUTPUT_FILE

# Schleife über alle Puffergrößen
for NX in "${BUFFER_SIZES[@]}"; do
    echo "Running benchmark for buffer size: $NX"

    # Serielle Version ausführen
    BASE_OUTPUT=$($BASE_EXEC $NX 2 10)
    BASE_TIME=$(echo "$BASE_OUTPUT" | grep "elapsed time" | awk '{print $3}')
    BASE_MLUP=$(echo "$BASE_OUTPUT" | grep "MLUP/s" | awk '{print $2}')
    BASE_BW=$(echo "$BASE_OUTPUT" | grep "bandwidth" | awk '{print $2}')
    echo "$NX, Serial, $BASE_TIME, $BASE_MLUP, $BASE_BW" >> $OUTPUT_FILE

    # OpenMP-Version ausführen
    OMP_OUTPUT=$($OMP_EXEC $NX 2 10)
    OMP_TIME=$(echo "$OMP_OUTPUT" | grep "elapsed time" | awk '{print $3}')
    OMP_MLUP=$(echo "$OMP_OUTPUT" | grep "MLUP/s" | awk '{print $2}')
    OMP_BW=$(echo "$OMP_OUTPUT" | grep "bandwidth" | awk '{print $2}')
    echo "$NX, OpenMP, $OMP_TIME, $OMP_MLUP, $OMP_BW" >> $OUTPUT_FILE

    # Cuda-Version ausführen
    CUDA_OUTPUT=$($CUDA_EXEC $NX 2 10)
    CUDA_TIME=$(echo "$CUDA_OUTPUT" | grep "elapsed time" | awk '{print $3}')
    CUDA_MLUP=$(echo "$CUDA_OUTPUT" | grep "MLUP/s" | awk '{print $2}')
    CUDA_BW=$(echo "$CUDA_OUTPUT" | grep "bandwidth" | awk '{print $2}')
    echo "$NX, CUDA, $CUDA_TIME, $CUDA_MLUP, $CUDA_BW" >> $OUTPUT_FILE

done

echo "Benchmark completed. Results saved to $OUTPUT_FILE."