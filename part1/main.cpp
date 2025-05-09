#include <iostream>
#include "knn_utils.h"
#include "feature_selector.h"

// Example main
int main() {
    // Ensure you have a "data.txt" file in the correct format and location
    // relative to where you run the executable.
    // Format: first column is class label, subsequent columns are features.
    // Example data.txt:
    // 1 0.5 1.2 0.8
    // 2 1.1 0.7 1.5
    // 1 0.3 1.0 0.9
    // ...

    std::cout << "Attempting to load data from 'data.txt'..." << std::endl;
    auto [X, y] = KNNUtils::loadData("data.txt");

    if (X.empty() || y.empty()) {
        std::cerr << "Error: Failed to load data or data file is empty." << std::endl;
        std::cerr << "Please ensure 'data.txt' exists in the same directory as the executable "
                  << "and is formatted correctly (label feature1 feature2 ...)." << std::endl;
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