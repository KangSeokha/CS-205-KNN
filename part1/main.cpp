#include <iostream>
#include <string>
#include <limits>
#include "knn_utils.h"
#include "feature_selector.h"

void printDatasetMenu() {
    std::cout << "\nAvailable datasets:" << std::endl;
    std::cout << "1. CS205 Large Dataset (CS205_large_Data__17.txt)" << std::endl;
    std::cout << "2. CS205 Small Dataset (CS205_small_Data__10.txt)" << std::endl;
    std::cout << "3. Diabetes Dataset (diabetes.csv)" << std::endl;
    std::cout << "Please enter your choice (1-3): ";
}

void printAlgorithmMenu() {
    std::cout << "\nAvailable algorithms:" << std::endl;
    std::cout << "1. Forward Selection" << std::endl;
    std::cout << "2. Backward Elimination" << std::endl;
    std::cout << "3. Both Algorithms" << std::endl;
    std::cout << "Please enter your choice (1-3): ";
}

void clearInputBuffer() {
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

int main() {
    // Get dataset choice
    int datasetChoice;
    std::string datasetFile;
    bool isCSV = false;

    while (true) {
        printDatasetMenu();
        if (!(std::cin >> datasetChoice)) {
            std::cout << "Invalid input. Please enter a number." << std::endl;
            clearInputBuffer();
            continue;
        }
        
        switch (datasetChoice) {
            case 1:
                datasetFile = "CS205_large_Data__17.txt";
                break;
            case 2:
                datasetFile = "CS205_small_Data__10.txt";
                break;
            case 3:
                datasetFile = "diabetes.csv";
                isCSV = true;
                break;
            default:
                std::cout << "Invalid choice. Please enter a number between 1 and 4." << std::endl;
                continue;
        }
        break;
    }

    // Load the selected dataset
    std::cout << "\nAttempting to load data from '" << datasetFile << "'..." << std::endl;
    std::pair<std::vector<std::vector<double>>, std::vector<int>> data;
    
    try {
        if (isCSV) {
            data = KNNUtils::loadCSVData(datasetFile);
        } else {
            data = KNNUtils::loadData(datasetFile);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        std::cerr << "Please ensure '" << datasetFile << "' exists in the same directory as the executable." << std::endl;
        return 1;
    }
    
    auto& X = data.first;
    auto& y = data.second;

    // Validate data
    if (X.empty() || y.empty()) {
        std::cerr << "Error: Failed to load data or data file is empty." << std::endl;
        std::cerr << "Please ensure '" << datasetFile << "' exists in the same directory as the executable "
                  << "and is formatted correctly." << std::endl;
        return 1;
    }
    if (X.size() != y.size()) {
        std::cerr << "Error: Mismatch between number of data samples and labels." << std::endl;
        return 1;
    }
    for(const auto& row : X) {
        if (row.empty()) {
            std::cerr << "Error: One of the data rows has no features." << std::endl;
            return 1;
        }
    }

    std::cout << "Data loaded successfully: " << X.size() << " samples, "
              << (X.empty() ? 0 : X[0].size()) << " features." << std::endl;

    // Get algorithm choice
    int algorithmChoice;
    while (true) {
        printAlgorithmMenu();
        if (!(std::cin >> algorithmChoice)) {
            std::cout << "Invalid input. Please enter a number." << std::endl;
            clearInputBuffer();
            continue;
        }
        
        if (algorithmChoice >= 1 && algorithmChoice <= 3) {
            break;
        }
        std::cout << "Invalid choice. Please enter a number between 1 and 3." << std::endl;
    }

    // Run selected algorithm(s)
    try {
        switch (algorithmChoice) {
            case 1:
                std::cout << "\nRunning Forward Selection..." << std::endl;
                FeatureSelector::forwardSelection(X, y);
                break;
            case 2:
                std::cout << "\nRunning Backward Elimination..." << std::endl;
                FeatureSelector::backwardElimination(X, y);
                break;
            case 3:
                std::cout << "\nRunning Forward Selection..." << std::endl;
                FeatureSelector::forwardSelection(X, y);
                std::cout << "\n----------------------------------------" << std::endl;
                std::cout << "----------------------------------------\n" << std::endl;
                std::cout << "Running Backward Elimination..." << std::endl;
                FeatureSelector::backwardElimination(X, y);
                break;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during algorithm execution: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}