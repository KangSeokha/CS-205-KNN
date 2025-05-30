#include "plot_utils.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>

void PlotUtils::writeDataFile(const std::vector<std::pair<std::vector<int>, double>>& results,
                            const std::string& dataFile) {
    std::ofstream file(dataFile);
    for (size_t i = 0; i < results.size(); ++i) {
        // Write feature set size and accuracy
        file << i + 1 << " " 
             << results[i].second * 100 << " \"";
        
        // Write feature set as label
        const auto& features = results[i].first;
        for (size_t j = 0; j < features.size(); ++j) {
            file << features[j] + 1;
            if (j < features.size() - 1) file << ",";
        }
        file << "\"\n";
    }
    file.close();
}

void PlotUtils::plotResults(const std::vector<std::pair<std::vector<int>, double>>& results,
                          const std::string& outputFile,
                          const std::string& title) {
    // Create temporary data file
    std::string dataFile = "temp_plot_data.txt";
    writeDataFile(results, dataFile);

    // Create gnuplot script
    std::ofstream script("plot_script.gp");
    script << "set terminal pngcairo enhanced font 'Arial,12' size 1200,800\n"
           << "set output '" << outputFile << "'\n"
           << "set title '" << title << "' font 'Arial,14'\n"
           << "set xlabel 'Number of Features' font 'Arial,12'\n"
           << "set ylabel 'Accuracy (%)' font 'Arial,12'\n"
           << "set grid\n"
           << "set key outside right\n"
           << "set style data histogram\n"
           << "set style fill solid 0.8\n"
           << "set boxwidth 0.8\n"
           << "plot '" << dataFile << "' using 2:xtic(3) title 'Feature Sets' with boxes lc rgb '#4169E1',\\\n"
           << "     '' using 0:2:2 with labels offset 0,1 notitle\n";
    script.close();

    // Execute gnuplot
    std::string command = "gnuplot plot_script.gp";
    int result = system(command.c_str());

    if (result == 0) {
        std::cout << "Plot saved as " << outputFile << std::endl;
        // Clean up temporary files
        remove(dataFile.c_str());
        remove("plot_script.gp");
    } else {
        std::cerr << "Error creating plot" << std::endl;
    }
} 