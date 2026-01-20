#!/bin/bash

gcc -o iterative_improvement iterative_improvement.c \
    -framework Accelerate \
    -O3 \
    -Wall

echo "Compilation complete"
