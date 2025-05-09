#ifndef FEATURE_SELECTOR_H
#define FEATURE_SELECTOR_H

#include <vector>
#include <string> // Though not directly used by methods, often included with vector
#include "knn_utils.h" // Needs KNNUtils for its operations

class FeatureSelector {
public:
    static void forwardSelection(const std::vector<std::vector<double> >& X, const std::vector<int>& y);
    static void backwardElimination(const std::vector<std::vector<double> >& X, const std::vector<int>& y);
};

#endif // FEATURE_SELECTOR_H