#include <iostream>
#include <queue>
#include <unordered_map>
#include <bitset>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>

#include <omp.h>
#include <cuda_runtime.h>

#define N 50000

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

        
        std::vector<std::string> splitInput(2);
        long long substrSize = input.size() / 3;
        splitInput[0] = input.substr(0, substrSize);
        splitInput[1] = input.substr(substrSize, input.size());

        std::mutex freqMutex;
        std::unordered_map<char, int> freqMap;
        int freq[256];

        std::thread cudaThread([this, &freq, &splitInput, &freqMutex, &freqMap]() {
            countCharsCUDA(splitInput[0], freq);
            for (int i = 0; i < 256; i++) {
                if (freq[i] > 0) {
                    freqMutex.lock();
                    freqMap[i] += freq[i];
                    freqMutex.unlock();
                }
            }
        });

        int num_threads = omp_get_max_threads();
        std::vector<std::string> splited = split_string(splitInput[1], num_threads);

        std::thread ompThread([this, &freqMap, &splitInput, &splited, num_threads, &freqMutex]() { 
            
            std::vector<std::unordered_map<char, int>> pqs(num_threads);
            #pragma omp parallel for
            for (int i = 0; i < splited.size(); i++) {
                pqs[i] = countCharsOMP(splited[i]);
            }

            for (int i = 0; i < pqs.size(); i++) {
                for (auto it = pqs[i].begin(); it != pqs[i].end(); it++) {
                    freqMutex.lock();
                    freqMap[it->first] += it->second;
                    freqMutex.unlock();
                }
            }
        });

        cudaThread.join();
        ompThread.join();
        
        for (auto it = freqMap.begin(); it != freqMap.end(); it++) {
            pq.push(Node(it->first, it->second));
        }

        createTree();
        createCode(&pq.top(), "");

        std::string encodedCUDA;
        std::string encodedOMP;

        std::thread cudaThread2([this, &encodedCUDA, &splitInput]() { 
            encodedCUDA = encodeCUDA(splitInput[0]);
        });

        std::thread ompThread2([this, &encodedOMP, &splited]() { 
            int num_threads = omp_get_max_threads();
            std::vector<std::string> encodedParts(num_threads);

            #pragma omp parallel for
            for (int i = 0; i < num_threads; i++) {
                encodedParts[i] = encodeOMP(splited[i]);
            }
            for (int i = 0; i < encodedParts.size(); i++) {
                encodedOMP += encodedParts[i];
            }
        });

        cudaThread2.join();
        ompThread2.join();

        //std::string encoded = encodedCUDA + encodedOMP;
        int j = 0;
        for (j = 0; encodedCUDA.length() % 8 != 0; j++) {
            encodedCUDA += encodedOMP[j];
        }
        encodedOMP.erase(0, j);

        int padding = 8 - encodedOMP.size() % 8;
        for (int i = 0; i < padding; i++)
            encodedOMP += "0";

        std::bitset<8> padBits(padding);
        encodedOMP += padBits.to_string();
        
        std::string result = "";
        for (int i = 0; i < encodedCUDA.length(); i += 8) {
            std::bitset<8> bits(encodedCUDA.substr(i, 8));
            result += static_cast<char>(bits.to_ulong());
        }
        for (int i = 0; i < encodedOMP.length(); i += 8) {
            std::bitset<8> bits(encodedOMP.substr(i, 8));
            result += static_cast<char>(bits.to_ulong());
        }
        return result;
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
    void countCharsCUDA(std::string input, int* freq) {
        NUM_BLOCKS = (input.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        int* d_freq;        
        cudaMalloc((void**)&d_freq, 256 * sizeof(int));
        cudaMemset(d_freq, 0, 256 * sizeof(int));

        charCountKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_input, input.size(), d_freq);

        cudaMemcpy(freq, d_freq, 256 * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_freq);
    }

    std::unordered_map<char, int> countCharsOMP(std::string input) {
        std::unordered_map<char, int> freqMap;
        for (int i = 0; i < input.size(); i++) 
            freqMap[input[i]]++;

        return freqMap;
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

    std::string encodeCUDA(std::string input) {
        int maxCodeLength = maxLengthOfCode();

        long long outputSize = input.length() * maxCodeLength;
        char* d_output;
        cudaMalloc(&d_output, outputSize * sizeof(char));
        cudaMemset(d_output, 0, outputSize * sizeof(char));

        char* d_huffmanCodes;
        cudaMalloc(&d_huffmanCodes, 256 * (maxCodeLength+1) * sizeof(char));
        cudaMemset(d_huffmanCodes, 0, 256 * (maxCodeLength+1) * sizeof(char));
    
        for (const auto& entry : code) {
            cudaMemcpy(d_huffmanCodes + entry.first * (maxCodeLength+1), entry.second.c_str(), entry.second.length() * sizeof(char), cudaMemcpyHostToDevice);
        }

        int blockDim = 32768;
        const int chunkSize = (input.size() + blockDim  - 1) / blockDim;

        encodeKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_input, input.length(), d_output, d_huffmanCodes, maxCodeLength, chunkSize);

        char* encoded = new char[outputSize]();
        cudaMemcpy((void*)encoded, d_output, outputSize * sizeof(char), cudaMemcpyDeviceToHost);

        std::string enc_str = "";
        for (int i = 0; i < outputSize; i++) 
            if (encoded[i] != '\0') 
                enc_str += encoded[i];

        delete[] encoded;
        cudaFree(d_output);
        cudaFree(d_huffmanCodes);
        cudaFree(d_input);
        
        return enc_str;
    }

    std::string encodeOMP(std::string input) {
        std::string encoded = "";
        for (int i = 0; i < input.length(); i++)
            encoded += code[input[i]];

        return encoded;
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
    

    std::vector<std::string> split_string(std::string content, int num) {
        int partSize = std::ceil((double) content.length() / num);
        std::vector<std::string> parts;
        for (int i = 0; i < num; i++) {
            int start = i * partSize;
            int end = std::min((i + 1) * partSize, (int) content.length());
            parts.push_back(content.substr(start, end - start));
            if (end == content.length())
                break;
        }
        return parts;
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

        THREADS_PER_BLOCK = prop.maxThreadsPerBlock;
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

    std::cout << "CUDA + OMP (" << input << "): " << elapsedMilliseconds / 1000 << " seconds" << std::endl;

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