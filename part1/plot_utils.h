#ifndef PLOT_UTILS_H
#define PLOT_UTILS_H

#include <vector>
#include <string>
#include <utility>

class PlotUtils {
public:
    static void plotResults(const std::vector<std::pair<std::vector<int>, double>>& results,
                          const std::string& outputFile,
                          const std::string& title);
private:
    static void writeDataFile(const std::vector<std::pair<std::vector<int>, double>>& results,
                            const std::string& dataFile);
};

#endif // PLOT_UTILS_H 