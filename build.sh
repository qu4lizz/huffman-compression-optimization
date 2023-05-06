#!/bin/bash

echo "Compiling huffman_compression.cpp"
g++ src/huffman_compression.cpp -o huffman_compression.exe

echo "Compiling huffman_compression_cuda.cu"
nvcc src/huffman_compression_cuda.cu -o huffman_compression_cuda.exe -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart

echo "Compiling huffman_compression_openmp.cpp"
g++ -fopenmp src/huffman_compression_openmp.cpp -o huffman_compression_openmp.exe

echo "Compiling hufman_compression_cuda_omp.cu"
nvcc src/huffman_compression_cuda_omp.cu -o huffman_compression_cuda_omp.exe -Xcompiler -fopenmp

echo "Compiling huffman_compression_o2.cpp"
g++ -O2 src/huffman_compression.cpp -o huffman_compression.exe

echo "Compiling huffman_compression_o3.cpp"
g++ -O3 src/huffman_compression.cpp -o huffman_compression.exe

echo "Compiling huffman_compression_cuda_o2.cu"
nvcc -O2 src/huffman_compression_cuda.cu -o huffman_compression_cuda.exe -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart

echo "Compiling huffman_compression_cuda_o3.cu"
nvcc -O3 src/huffman_compression_cuda.cu -o huffman_compression_cuda.exe -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart

echo "Compiling huffman_compression_openmp_o2.cpp"
g++ -O2 -fopenmp src/huffman_compression_openmp.cpp -o huffman_compression_openmp.exe

echo "Compiling huffman_compression_openmp_o3.cpp"
g++ -O3 -fopenmp src/huffman_compression_openmp.cpp -o huffman_compression_openmp.exe

echo "Compiling hufman_compression_cuda_omp_o2.cu"
nvcc -O2 src/huffman_compression_cuda_omp.cu -o huffman_compression_cuda_omp.exe -Xcompiler -fopenmp

echo "Compiling hufman_compression_cuda_omp_o3.cu"
nvcc -O3 src/huffman_compression_cuda_omp.cu -o huffman_compression_cuda_omp.exe -Xcompiler -fopenmp
