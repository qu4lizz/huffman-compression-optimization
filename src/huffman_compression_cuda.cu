#include <iostream>
#include <queue>
#include <unordered_map>
#include <bitset>
#include <fstream>
#include <string>
#include <chrono>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define N 60000

int THREADS_PER_BLOCK;
int NUM_BLOCKS;

__global__ void charCountKernel(const char* input, const int length, int* freq) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 0 && i < length) {
        int index = input[i];
        if (index < 0) {
            printf("Can't use Non-ASCII characters.\n");
            return;
        }
        atomicAdd(&freq[index], 1);
    }
}

__global__ void encodeKernel(const char* d_input, const int inputSize, char* d_output, char* d_huffmanCodes, const int maxCodeLength, const int chunkSize) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long start = tid * chunkSize;
    if (start >= inputSize && start < 0) return;
    long long end = start + chunkSize > inputSize ? inputSize : start + chunkSize;
    
    // Encode each part
    int encodedSize = 0;
    if (N < chunkSize * 8) {
        printf("N is too small. Increase N(%d) to at least %d.\n", N, chunkSize * 8);
        return;
    }
    char encoded[N]; 
    long long i = start;
    while (i < end) {
        char c = d_input[i];
        char* code = d_huffmanCodes + c * (maxCodeLength+1);
        int codeLength = 0;
        while (code[codeLength] != '\0') {
            codeLength++;
        }
        for (int j = 0; j < codeLength; j++) {
            encoded[encodedSize++] = code[j];
        }
        i++;
    }

    long long outputStart = tid * chunkSize * maxCodeLength;

    for (int i = 0; i < encodedSize; i++) {
        d_output[outputStart + i] = encoded[i];
    }
}

class HuffmanCompression {
private:
    struct Node {
        char ch;
        int freq;
        Node *left, *right;
        
        Node(char ch, int freq) {
            this->ch = ch;
            this->freq = freq;
            this->left = this->right = nullptr;
        }

        bool operator<(const Node& other) const {
            return freq > other.freq;
        }
    };

    std::string input;
    std::priority_queue<Node> pq;
    std::unordered_map<char, std::string> code;
    int freq[256];
    char* d_input;


public:
    HuffmanCompression(std::string input) {
        this->input = input;
    }

    std::priority_queue<Node>& getPq() {
        return pq;
    }

    std::unordered_map<char, std::string>& getCode() {
        return code;
    }

    std::string compress() {
        cudaMalloc((void**)&d_input, input.size() * sizeof(char));
        cudaMemset(d_input, 0, input.size() * sizeof(char));
        cudaMemcpy(d_input, input.c_str(), input.size() * sizeof(char), cudaMemcpyHostToDevice);

        countChars();

        createTree();
        createCode(&pq.top(), "");


        return encode();
    }

    std::string decompress(const std::string& encoded) {

        // Convert the encoded string to a binary string.
        std::string binary_pad = "";

        for (int i = 0; i < encoded.length(); i++) {
            std::bitset<8> bits(encoded[i]);
            binary_pad += bits.to_string();
        }

        // Convert the padded bits at the end to an integer.
        std::string padding_str = binary_pad.substr(binary_pad.length() - 8);
        int padding = std::bitset<8>(padding_str).to_ulong();

        // Remove the padded bits from the encoded string.
        std::string binary = binary_pad.substr(0, binary_pad.length() - 8 - padding);

        
        std::unordered_map<std::string, char> decodeMap;
        for (const auto& p : code) {
            decodeMap[p.second] = p.first;
        }

        std::string decoded = "";
        std::string temp = "";
        for (int i = 0; i < binary.length(); i++) {
            temp += binary[i];
            if (decodeMap.find(temp) != decodeMap.end()) {
                decoded += decodeMap[temp];
                temp = "";
            }
        }

        return decoded;
    }

private:
    void countChars() {
        NUM_BLOCKS = (input.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        int* d_freq;        
        cudaMalloc((void**)&d_freq, 256 * sizeof(int));
        cudaMemset(d_freq, 0, 256 * sizeof(int));

        charCountKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_input, input.size(), d_freq);

        cudaMemcpy(freq, d_freq, 256 * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_freq);

        for (int i = 0; i < 256; i++) {
            if (freq[i] > 0) {
                pq.emplace(i, freq[i]);
            }
        }
    }

    void createTree() {
        while (pq.size() > 1) {
            Node* left = new Node(pq.top());
            pq.pop();
            Node* right = new Node(pq.top());
            pq.pop();
            Node* parent = new Node('\0', left->freq + right->freq);
            parent->left = left;
            parent->right = right;
            pq.push(*parent);
        }
    }

    void createCode(const Node* node, std::string s) {
        if (!node)
            return;

        if (node->ch)
            code[node->ch] = s;

        createCode(node->left, s + "0");
        createCode(node->right, s + "1");
    }

    std::string encode() {
        int maxCodeLength = maxLengthOfCode();

        long long outputSize = input.length() * maxCodeLength;
        char* d_output;
        cudaMalloc(&d_output, outputSize * sizeof(char));
        cudaMemset(d_output, 0, outputSize * sizeof(char));

        char* d_huffmanCodes;
        cudaMalloc(&d_huffmanCodes, 256 * (maxCodeLength+1) * sizeof(char));
        cudaMemset(d_huffmanCodes, 0, 256 * (maxCodeLength+1) * sizeof(char));
    
        for (const auto& entry : code)
            cudaMemcpy(d_huffmanCodes + entry.first * (maxCodeLength+1), entry.second.c_str(), entry.second.length() * sizeof(char), cudaMemcpyHostToDevice);

        int blockDim = 32768;
        const int chunkSize = (input.size() + blockDim  - 1) / blockDim;

        encodeKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_input, input.length(), d_output, d_huffmanCodes, maxCodeLength, chunkSize);
        cudaDeviceSynchronize();
        
        cudaPeekAtLastError();

        char* encoded = new char[outputSize]();
        cudaMemcpy((void*)encoded, d_output, outputSize * sizeof(char), cudaMemcpyDeviceToHost);

        std::string enc_str = "";
        long long i = 0;
        while (i < outputSize) {
            if (encoded[i] != '\0') 
                enc_str += encoded[i];
            i++;
        }

        delete[] encoded;
        cudaFree(d_output);
        cudaFree(d_huffmanCodes);
        cudaFree(d_input);

        int padding = 8 - enc_str.length() % 8;
        for (int i = 0; i < padding; i++)
            enc_str += "0";

        std::bitset<8> padBits(padding);
        enc_str += padBits.to_string();
    
        std::string result = "";

        for (int i = 0; i < enc_str.length(); i += 8) {
            std::bitset<8> bits(enc_str.substr(i, 8));
            result += static_cast<char>(bits.to_ulong());
        }

        return result;
    }

    int maxLengthOfCode() {
        int max = 0;
        for (const auto& entry : code) {
            if (entry.second.length() > max) {
                max = entry.second.length();
            }
        }
        return max;
    }
};

int main(int argc, char** argv) {
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    else {
        int device;
        cudaGetDevice(&device);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        THREADS_PER_BLOCK = 1024;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    std::string input;
    std::string output;
    if (argc != 3) {
        std::cout << "Usage: ./huffman_compression_openmp <input file> <output file>\n";
        std::cout << "using default files\n";
        input = "resources/default.txt";
        output = "resources/default.cuda";
    }
    else {
        input = argv[1];
        output = argv[2];
    }

    std::ifstream inFile(input, std::ios::binary);
    if (!inFile.is_open()) {
        std::cout << "Failed to open input file\n";
        return 0;
    }
    std::string fileContent((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());

    HuffmanCompression hc(fileContent);
    std::string encoded = hc.compress();

    std::ofstream outFile(output, std::ios::binary);
    if (!outFile.is_open()) {
        std::cout << "Failed to open output file\n";
        return 0;
    }

    outFile.write(encoded.c_str(), encoded.length());

    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float elapsedMilliseconds = 0;
    cudaEventElapsedTime(&elapsedMilliseconds, start, stop);

    std::cout << "CUDA (" << input << "): " << elapsedMilliseconds / 1000 << " seconds" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    /*std::string decoded = hc.decompress(encoded);
    if (decoded == fileContent) {
        std::cout << "Compression successful\n";
    }
    else {
        std::cout << "Compression failed\n";
    }*/

    return 0;
}