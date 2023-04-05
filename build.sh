#!/bin/bash

echo "Compiling huffman_compression.cpp"
g++ src/huffman_compression.cpp -o huffman_compression.exe

echo "Compiling huffman_compression_cuda.cu"
nvcc src/huffman_compression_cuda.cu -o huffman_compression_cuda.exe -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart

echo "Compiling huffman_compression_openmp.cpp"
g++ -fopenmp src/huffman_compression_openmp.cpp -o huffman_compression_openmp.exe

echo "Compiling hufman_compression_cuda_omp.cu"
nvcc src/huffman_compression_cuda_omp.cu -o huffman_compression_cuda_omp.exe -Xcompiler -fopenmp
