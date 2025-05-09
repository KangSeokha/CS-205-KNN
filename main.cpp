// feature_selection.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <limits>

class KNNUtils {
public:
    static std::pair<std::vector<std::vector<double> >, std::vector<int> > loadData(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        std::vector<std::vector<double> > X;
        std::vector<int> y;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<double> row;
            double value;
            bool isFirst = true;
            int label;
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
        return make_pair(X, y);
    }

    static std::vector<double> zNormalize(const std::vector<double>& data) {
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / data.size() - mean * mean);

        std::vector<double> normalized;
        for (double d : data) {
            normalized.push_back((d - mean) / stdev);
        }
        return normalized;
    }

    static double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
        double dist = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            dist += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(dist);
    }

    static double nnLeaveOneOutCV(const std::vector<std::vector<double> >& X, const std::vector<int>& y) {
        int correct = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            double minDist = std::numeric_limits<double>::max();
            int predicted = -1;
            for (size_t j = 0; j < X.size(); ++j) {
                if (i == j) continue;
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
};

class FeatureSelector {
public:
    static void forwardSelection(const std::vector<std::vector<double> >& X, const std::vector<int>& y) {
        size_t numFeatures = X[0].size();
        std::vector<int> allFeatures(numFeatures);
        std::iota(allFeatures.begin(), allFeatures.end(), 0);
        std::vector<int> selectedFeatures;

        double globalBestAcc = -1.0;
        std::vector<int> bestFeatures;

        std::cout << "Beginning forward selection.\n";

        while (!allFeatures.empty()) {
            std::cout << "\nEvaluating features with current selected set:";
            for (int sf : selectedFeatures) std::cout << " " << sf;
            std::cout << "\n";

            double bestLocalAcc = -1.0;
            int bestFeature = -1;
            for (int f : allFeatures) {
                std::vector<int> trialFeatures = selectedFeatures;
                trialFeatures.push_back(f);

                std::vector<std::vector<double> > projectedX;
                for (const auto& row : X) {
                    std::vector<double> proj;
                    for (int idx : trialFeatures) {
                        proj.push_back(row[idx]);
                    }
                    projectedX.push_back(proj);
                }
                double acc = KNNUtils::nnLeaveOneOutCV(projectedX, y);
                std::cout << "    Trying feature " << f << " results in accuracy " << acc * 100 << "%\n";
                if (acc > bestLocalAcc) {
                    bestLocalAcc = acc;
                    bestFeature = f;
                }
            }
            selectedFeatures.push_back(bestFeature);
            allFeatures.erase(std::remove(allFeatures.begin(), allFeatures.end(), bestFeature), allFeatures.end());

            std::cout << "Selected feature " << bestFeature << " with local best accuracy: " << bestLocalAcc * 100 << "%\n";

            if (bestLocalAcc > globalBestAcc) {
                globalBestAcc = bestLocalAcc;
                bestFeatures = selectedFeatures;
            }
        }

        std::cout << "\nFinished search!! The best feature subset is:";
        for (int f : bestFeatures) std::cout << " " << f;
        std::cout << ", which has accuracy of " << globalBestAcc * 100 << "%\n";
    }

    static void backwardElimination(const std::vector<std::vector<double> >& X, const std::vector<int>& y) {
        size_t numFeatures = X[0].size();
        std::vector<int> allFeatures(numFeatures);
        std::iota(allFeatures.begin(), allFeatures.end(), 0);
        std::vector<int> currentCandidates = allFeatures;

        std::cout << "Calculating initial global best accuracy\n";
        double globalBestAcc = KNNUtils::nnLeaveOneOutCV(X, y) * 100;
        std::vector<int> globalBestFeatures = allFeatures;
        std::cout << "Global best accuracy using all " << numFeatures << " features is " << globalBestAcc << "%\n";

        std::cout << "Beginning search.\n";
        while (!currentCandidates.empty()) {
            std::cout << std::endl;
            double localBestAcc = -std::numeric_limits<double>::infinity();
            std::vector<int> localBestFeatures;
            int featureToRemove = -1;

            for (int f : currentCandidates) {
                std::vector<int> trialFeatures = currentCandidates;
                trialFeatures.erase(std::remove(trialFeatures.begin(), trialFeatures.end(), f), trialFeatures.end());

                std::vector<std::vector<double> > projectedX;
                for (const auto& row : X) {
                    std::vector<double> proj;
                    for (int idx : trialFeatures) {
                        proj.push_back(row[idx]);
                    }
                    projectedX.push_back(proj);
                }

                double acc = KNNUtils::nnLeaveOneOutCV(projectedX, y) * 100;
                std::cout << "    Using feature(s)";
                for (int tf : trialFeatures) std::cout << " " << tf;
                std::cout << " accuracy is " << acc << "%\n";

                if (acc > localBestAcc) {
                    localBestAcc = acc;
                    featureToRemove = f;
                    localBestFeatures = trialFeatures;
                }
            }

            std::cout << std::endl;
            if (localBestAcc < globalBestAcc) {
                std::cout << "(WARNING, Accuracy has decreased! Continuing search in case of local maximum)\n";
            } else {
                globalBestAcc = localBestAcc;
                globalBestFeatures = localBestFeatures;
            }

            std::cout << "Feature set";
            for (int f : localBestFeatures) std::cout << " " << f;
            std::cout << " was best, accuracy is " << localBestAcc << "%\n";

            currentCandidates = localBestFeatures;
        }

        std::cout << "\nFinished search!! The best feature subset is";
        for (int f : globalBestFeatures) std::cout << " " << f;
        std::cout << ", which has accuracy of " << globalBestAcc << "%\n";
    }
};

// Example main
int main() {
    auto [X, y] = KNNUtils::loadData("data.txt");
    FeatureSelector::forwardSelection(X, y);
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    FeatureSelector::backwardElimination(X, y);
    return 0;
}
