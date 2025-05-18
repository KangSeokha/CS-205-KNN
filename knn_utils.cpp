#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include <numeric>
#include <cmath>
#include <limits>

class KNNUtils {
public:
    std::pair<std::vector<std::vector<double> >, std::vector<int> > loadCSVData(const std::string& filename);
    std::pair<std::vector<std::vector<double> >, std::vector<int> > loadData(const std::string& filename);
    std::vector<double> zNormalize(const std::vector<double>& data);
    double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b);
    double nnLeaveOneOutCV(const std::vector<std::vector<double> >& X, const std::vector<int>& y);
};

std::pair<std::vector<std::vector<double> >, std::vector<int> > KNNUtils::loadCSVData(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<double> > X;
    std::vector<int> y;

    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        int feature_count = 0;

        while (std::getline(ss, value, ',')) {
            double num_value = std::stod(value);
            if (feature_count < 8) { // First 8 columns are features
                row.push_back(num_value);
            } else { // Last column is the label
                y.push_back(static_cast<int>(num_value));
            }
            feature_count++;
        }
        
        if (!row.empty()) {
            X.push_back(row);
        }
    }
    return std::make_pair(X, y);
}

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
        int label = 0;
        while (ss >> value) {
            if (isFirst) {
                label = static_cast<int>(value);
                isFirst = false;
            } else {
                row.push_back(value);
            }
        }
        if (!row.empty()) {
            X.push_back(row);
            y.push_back(label);
        }
    }
    return std::make_pair(X, y);
}

std::vector<double> KNNUtils::zNormalize(const std::vector<double>& data) {
    if (data.empty()) {
        return {};
    }
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / data.size() - mean * mean);

    std::vector<double> normalized;
    if (stdev == 0) {
        normalized.resize(data.size(), 0.0);
    } else {
        for (double d : data) {
            normalized.push_back((d - mean) / stdev);
        }
    }
    return normalized;
}

double KNNUtils::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double dist = 0;
    size_t size = std::min(a.size(), b.size());
    for (size_t i = 0; i < size; ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(dist);
}

double KNNUtils::nnLeaveOneOutCV(const std::vector<std::vector<double> >& X, const std::vector<int>& y) {
    if (X.empty() || X.size() != y.size()) {
        return 0.0;
    }
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        double minDist = std::numeric_limits<double>::max();
        int predicted = -1;
        for (size_t j = 0; j < X.size(); ++j) {
            if (i == j) continue;
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