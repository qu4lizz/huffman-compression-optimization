# Algorithm optimization - Huffman compression

## Description

<p align="justify"> The Huffman coding algorithm is a lossless data compression algorithm. It works by assigning variable-length codes to each symbol based on their frequency of occurrence in the input text. More frequent symbols get shorter codes, while less frequent ones get longer ones, so the resulting compressed text is a more efficient representation of the input text. </p>

## Usage

<p align="justify"> The build.sh script is used to compile the source code of all attached source code files. The executable file without optimization is compiled without the compiler optimization flag, so it uses the -O0 default optimization flag. </p>

<p align="justify"> The run.sh script is used to run all executables. It takes as arguments the input file and the name of the output file without the extension, because it assigns the extension depending on which executable is launched: 
<ul>
    <li>.out - no optimization</li>
    <li>.cuda - CUDA</li>
    <li>.omp - OpenMP</li>
    <li>.cuda_omp - a combination of CUDA and OpenMP.</li>
</ul>
If no arguments are passed, it uses default file.
</p>