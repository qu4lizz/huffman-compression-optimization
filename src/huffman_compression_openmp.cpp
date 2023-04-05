#include <iostream>
#include <queue>
#include <unordered_map>
#include <bitset>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <cmath>
#include <omp.h>

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

        void setFreq(int freq) {
            this->freq = freq;
        }

        bool operator<(const Node& other) const {
            return freq > other.freq;
        }
        bool operator==(const Node& other) const {
            return ch == other.ch;
        }
    };

    std::string input;
    std::priority_queue<Node> pq;
    std::unordered_map<char, std::string> code;

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
        int num_threads = omp_get_max_threads();

        // Split the input string into parts
        std::vector<std::string> splited = split_string(input, num_threads);

        // Count the frequency of each character in each part
        std::vector<std::unordered_map<char, int>> pqs(num_threads);
        #pragma omp parallel for
        for (int i = 0; i < splited.size(); i++) {
            pqs[i] = countChars(splited[i]);
        }

        // Merge the priority queues where same characters are merged into same node, but with the sum of their frequencies
        std::unordered_map<char, int> freqMap;
        for (int i = 0; i < pqs.size(); i++) {
            for (auto it = pqs[i].begin(); it != pqs[i].end(); it++) {
                freqMap[it->first] += it->second;
            }
        }
        for (const auto& p : freqMap) {
            pq.emplace(p.first, p.second);
        }
        

        // Create Huffman tree
        createTree();
        // Create Huffman codes
        createCode(&pq.top(), "");

        // Encode each part
        std::vector<std::string> encoded_parts(splited.size());

        #pragma omp parallel for
        for (int i = 0; i < splited.size(); i++) {
            encoded_parts[i] = encode(splited[i]);
        }
        std::string encoded = "";
        for (int i = 0; i < encoded_parts.size(); i++) {
            encoded += encoded_parts[i];
        }
        
        int padding = 8 - encoded.length() % 8;
        for (int i = 0; i < padding; i++) {
            encoded += "0";
        }

        std::bitset<8> padBits(padding);
        encoded += padBits.to_string();
        
    
        std::string result = "";
        for (int i = 0; i < encoded.length(); i += 8) {
            std::bitset<8> bits(encoded.substr(i, 8));
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
    std::unordered_map<char, int> countChars(std::string input) {
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

    std::string encode(std::string input) {
        std::string encoded = "";
        for (int i = 0; i < input.length(); i++) {
            encoded += code[input[i]];
        }

        return encoded;
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
    double start_time = omp_get_wtime();
    std::string input;
    std::string output;
    if (argc != 3) {
        std::cout << "Usage: ./huffman_compression_openmp <input file> <output file>\n";
        std::cout << "using default files\n";
        input = "resources/default.txt";
        output = "resources/default.omp";
    }
    else {
        input = argv[1];
        output = argv[2];
    }

    std::ifstream infile(input, std::ios::binary);
    if (!infile.is_open()) {
        std::cout << "Failed to open input file\n";
        return 0;
    }
    std::string fileContent((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());

    HuffmanCompression hc(fileContent);
    std::string encoded = hc.compress();

    std::ofstream outfile(output, std::ios::binary);
    if (!outfile.is_open()) {
        std::cout << "Failed to open output file\n";
        return 0;
    }

    outfile.write(encoded.c_str(), encoded.length());

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    std::cout << "OpenMP (" << input << "): " << elapsed_time << " seconds\n";
    /*std::string decoded = hc.decompress(encoded);
    if (decoded == fileContent) {
        std::cout << "Compression successful\n";
    }
    else {
        std::cout << "Compression failed\n";
    }*/
    return 0;
}

