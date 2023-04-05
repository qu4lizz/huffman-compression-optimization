#!/bin/bash

# Set input and output filenames based on arguments
input_file="$1"
output_file="$2"

# Call huffman_compression with input and output filenames as arguments
./huffman_compression.exe "$input_file" "$output_file.out"
echo "Compression: $(echo "scale=2; $(stat -c '%s' $output_file.out) / $(stat -c '%s' $input_file)" | bc -l)%"
echo ''

# Call huffman_compression_cuda with input and output filenames as arguments
./huffman_compression_cuda.exe "$input_file" "$output_file.cuda"
echo "Compression: $(echo "scale=2; $(stat -c '%s' $output_file.cuda) / $(stat -c '%s' $input_file)" | bc -l)%"
echo ''

# Call huffman_compression_openmp with input and output filenames as arguments
./huffman_compression_openmp.exe "$input_file" "$output_file.omp"
echo "Compression: $(echo "scale=2; $(stat -c '%s' $output_file.omp) / $(stat -c '%s' $input_file)" | bc -l)%"
echo ''

# Call huffman_compression_cuda_omp with input and output filenames as arguments
./huffman_compression_cuda_omp.exe "$input_file" "$output_file.cuda_omp"
echo "Compression: $(echo "scale=2; $(stat -c '%s' $output_file.cuda_omp) / $(stat -c '%s' $input_file)" | bc -l)%"
echo ''
