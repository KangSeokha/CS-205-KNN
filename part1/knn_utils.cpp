#include "knn_utils.h"
#include <fstream>
#include <sstream>
#include <numeric> // For std::accumulate, std::iota, std::inner_product
#include <cmath>   // For std::sqrt
#include <limits>  // For std::numeric_limits

std::pair<std::vector<std::vector<double> >, std::vector<int> > KNNUtils::loadData(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<double> > X;
    std::vector<int> y;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;
        bool isFirst = true;
        int label = 0; // Initialize label
        while (ss >> value) {
            if (isFirst) {
                label = static_cast<int>(value);
                isFirst = false;
            } else {
                row.push_back(value);
            }
        }
        X.push_back(row);
        y.push_back(label);
    }
    return std::make_pair(X, y);
}

std::vector<double> KNNUtils::zNormalize(const std::vector<double>& data) {
    if (data.empty()) {
        return {}; // Return empty if data is empty to avoid division by zero
    }
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / data.size() - mean * mean);

    std::vector<double> normalized;
    if (stdev == 0) { // Handle case where standard deviation is zero
        for (size_t i = 0; i < data.size(); ++i) {
            normalized.push_back(0.0); // Or data[i] - mean, depending on desired behavior
        }
    } else {
        for (double d : data) {
            normalized.push_back((d - mean) / stdev);
        }
    }
    return normalized;
}

double KNNUtils::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double dist = 0;
    // Ensure vectors are of the same size to avoid out-of-bounds access
    size_t size = std::min(a.size(), b.size());
    for (size_t i = 0; i < size; ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(dist);
}

double KNNUtils::nnLeaveOneOutCV(const std::vector<std::vector<double> >& X, const std::vector<int>& y) {
    if (X.empty() || X.size() != y.size()) {
        return 0.0; // Or handle error appropriately
    }
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        double minDist = std::numeric_limits<double>::max();
        int predicted = -1;
        for (size_t j = 0; j < X.size(); ++j) {
            if (i == j) continue;
            // Ensure X[i] and X[j] are not empty before calculating distance
            if (X[i].empty() || X[j].empty()) continue;
            double dist = euclideanDistance(X[i], X[j]);
            if (dist < minDist) {
                minDist = dist;
                predicted = y[j];
            }
        }
        if (predicted == y[i]) correct++;
    }
    return static_cast<double>(correct) / X.size();
}