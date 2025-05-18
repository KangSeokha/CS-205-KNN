#include <iostream>
#include "knn_utils.h"
#include "feature_selector.h"

int main() {
    std::cout << "Attempting to load data from 'diabetes.csv'..." << std::endl;
    auto [X, y] = KNNUtils::loadCSVData("diabetes.csv");

    if (X.empty() || y.empty()) {
        std::cerr << "Error: Failed to load data or data file is empty." << std::endl;
        std::cerr << "Please ensure 'diabetes.csv' exists in the same directory as the executable "
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
              << (X.empty() ? 0 : X[0].size()) << " features." << std::endl << std::endl;

    FeatureSelector::forwardSelection(X, y);
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "----------------------------------------\n" << std::endl;
    FeatureSelector::backwardElimination(X, y);

    return 0;
} 