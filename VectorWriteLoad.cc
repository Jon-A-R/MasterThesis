#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <sstream>
// Function to write a vector of doubles to a text file
void writeVectorToFile(const std::string& filename, const std::vector<double>& data) {
    std::ofstream outputFile(filename);

    if (outputFile.is_open()) {
        std::copy(data.begin(), data.end(), std::ostream_iterator<double>(outputFile, " "));
    }

    outputFile.close();
}

// Function to read a vector of doubles from a text file
void readVectorFromFile(const std::string& filename, std::vector<double>& data) {
    std::ifstream inputFile(filename);

    if (inputFile.is_open()) {
        std::string line;
        std::getline(inputFile, line);

        std::istringstream iss(line);
        double value;
        while (iss >> value) {
            data.push_back(value);
        }
    }

    inputFile.close();
}
