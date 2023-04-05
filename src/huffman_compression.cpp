#include <iostream>
#include <queue>
#include <unordered_map>
#include <bitset>
#include <fstream>
#include <string>
#include <chrono>


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
        std::unordered_map<char, int> freqMap;
        for (char c : input) {
            freqMap[c]++;
        }
        for (const auto& p : freqMap) {
            pq.emplace(p.first, p.second);
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
        std::string encoded = "";
        for (char c : input) {
            encoded += code[c];
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
};

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    std::string input;
    std::string output;
    if (argc != 3) {
        std::cout << "Usage: ./huffman_compression <input file> <output file>\n";
        std::cout << "using default files\n";
        input = "resources/default.txt";
        output = "resources/default.out";
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

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "No optimization (" << input << "): " << (double)duration.count() / 1000000 << " seconds" << std::endl;
    
    /*std::string decoded = hc.decompress(encoded);
    if (decoded == fileContent) {
        std::cout << "Compression successful\n";
    }
    else {
        std::cout << "Compression failed\n";
    }*/
    return 0;
}