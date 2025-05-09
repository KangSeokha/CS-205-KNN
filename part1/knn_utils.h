#ifndef KNN_UTILS_H
#define KNN_UTILS_H

#include <vector>
#include <string>
#include <utility> // For std::pair

class KNNUtils {
public:
    static std::pair<std::vector<std::vector<double> >, std::vector<int> > loadData(const std::string& filename);
    static std::vector<double> zNormalize(const std::vector<double>& data);
    static double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b);
    static double nnLeaveOneOutCV(const std::vector<std::vector<double> >& X, const std::vector<int>& y);
};

#endif // KNN_UTILS_H