#ifndef FEATURE_SELECTOR_H
#define FEATURE_SELECTOR_H

#include <vector>
#include <string> // Though not directly used by methods, often included with vector
#include <utility>
#include "knn_utils.h" // Needs KNNUtils for its operations

class FeatureSelector {
public:
    static std::vector<std::pair<std::vector<int>, double>> forwardSelection(
        const std::vector<std::vector<double> >& X, 
        const std::vector<int>& y
    );
    
    static std::vector<std::pair<std::vector<int>, double>> backwardElimination(
        const std::vector<std::vector<double> >& X, 
        const std::vector<int>& y
    );
};

#endif // FEATURE_SELECTOR_H