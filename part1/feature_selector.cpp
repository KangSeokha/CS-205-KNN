#include "feature_selector.h"
#include "knn_utils.h" // Already included in .h, but good practice for .cpp if directly using its types
#include <iostream>
#include <algorithm> // For std::remove, std::iota
#include <numeric>   // For std::iota (though already in knn_utils.cpp, this makes this unit more self-contained if needed)
#include <limits>    // For std::numeric_limits

void FeatureSelector::forwardSelection(const std::vector<std::vector<double> >& X, const std::vector<int>& y) {
    if (X.empty() || X[0].empty()) {
        std::cout << "Input data X is empty or has no features. Aborting forward selection." << std::endl;
        return;
    }
    size_t numFeatures = X[0].size();
    std::vector<int> allFeatures(numFeatures);
    std::iota(allFeatures.begin(), allFeatures.end(), 0); // 0, 1, 2, ..., numFeatures-1
    std::vector<int> selectedFeatures;

    double globalBestAcc = -1.0;
    std::vector<int> bestFeaturesOverall; // Renamed to avoid conflict

    std::cout << "Beginning forward selection.\n";

    for (size_t k = 0; k < numFeatures; ++k) { // Iterate up to numFeatures times
        std::cout << "\nOn level " << k + 1 << " of the search tree\n";
        std::cout << "Current selected feature set: {";
        for (size_t i = 0; i < selectedFeatures.size(); ++i) {
            std::cout << selectedFeatures[i] + 1 << (i == selectedFeatures.size() - 1 ? "" : ", ");
        }
        std::cout << "}\n";

        double bestLocalAcc = -1.0;
        int featureToAddThisLevel = -1;

        for (int f_idx = 0; f_idx < allFeatures.size(); ++f_idx) {
            int featureToConsider = allFeatures[f_idx];
            // Check if feature is already selected (it shouldn't be if allFeatures is managed correctly)
            // This check is more for safety if logic changes:
            // if (std::find(selectedFeatures.begin(), selectedFeatures.end(), featureToConsider) != selectedFeatures.end()) {
            //    continue;
            // }

            std::vector<int> trialFeatures = selectedFeatures;
            trialFeatures.push_back(featureToConsider);
            std::sort(trialFeatures.begin(), trialFeatures.end()); // Keep sorted for consistent output

            std::vector<std::vector<double> > projectedX;
            for (const auto& row : X) {
                std::vector<double> proj;
                for (int idx : trialFeatures) {
                    proj.push_back(row[idx]);
                }
                projectedX.push_back(proj);
            }
            double acc = KNNUtils::nnLeaveOneOutCV(projectedX, y);
            std::cout << "    Considering adding feature " << featureToConsider + 1 << " with current set {";
            for(size_t i = 0; i < trialFeatures.size(); ++i) {
                std::cout << trialFeatures[i] + 1 << (i == trialFeatures.size() - 1 ? "" : ", ");
            }
            std::cout << "} accuracy is " << acc * 100 << "%\n";

            if (acc > bestLocalAcc) {
                bestLocalAcc = acc;
                featureToAddThisLevel = featureToConsider;
            }
        }

        if (featureToAddThisLevel != -1) {
            selectedFeatures.push_back(featureToAddThisLevel);
            std::sort(selectedFeatures.begin(), selectedFeatures.end());
            // Remove the chosen feature from allFeatures
            allFeatures.erase(std::remove(allFeatures.begin(), allFeatures.end(), featureToAddThisLevel), allFeatures.end());

            std::cout << "\nOn level " << k + 1 << ", added feature " << featureToAddThisLevel + 1 << " to current set. Accuracy: " << bestLocalAcc * 100 << "%\n";
            std::cout << "Current best feature set: {";
            for(size_t i = 0; i < selectedFeatures.size(); ++i) {
                std::cout << selectedFeatures[i] + 1 << (i == selectedFeatures.size() - 1 ? "" : ", ");
            }
            std::cout << "} with accuracy " << bestLocalAcc * 100 << "%\n";


            if (bestLocalAcc > globalBestAcc) {
                globalBestAcc = bestLocalAcc;
                bestFeaturesOverall = selectedFeatures;
            } else {
                 std::cout << "(Warning, accuracy has decreased or stayed the same. Global best is still better.)\n";
            }
        } else {
            std::cout << "\nNo feature improved accuracy at this level. Halting forward selection.\n";
            break; // No improvement possible
        }
         if (selectedFeatures.size() == numFeatures) break; // All features selected
    }

    std::cout << "\nFinished forward selection!! The best feature subset is: {";
    for (size_t i = 0; i < bestFeaturesOverall.size(); ++i) {
        std::cout << bestFeaturesOverall[i] + 1 << (i == bestFeaturesOverall.size() - 1 ? "" : ", ");
    }
    std::cout << "}, which has an accuracy of " << globalBestAcc * 100 << "%\n";
}


void FeatureSelector::backwardElimination(const std::vector<std::vector<double> >& X, const std::vector<int>& y) {
    if (X.empty() || X[0].empty()) {
        std::cout << "Input data X is empty or has no features. Aborting backward elimination." << std::endl;
        return;
    }
    size_t numFeatures = X[0].size();
    std::vector<int> currentFeatures(numFeatures);
    std::iota(currentFeatures.begin(), currentFeatures.end(), 0); // 0, 1, ..., numFeatures-1

    std::cout << "Calculating initial accuracy with all features.\n";
    double globalBestAcc = KNNUtils::nnLeaveOneOutCV(X, y);
    std::vector<int> bestFeaturesOverall = currentFeatures;

    std::cout << "Initial feature set: {";
    for(size_t i = 0; i < currentFeatures.size(); ++i) {
        std::cout << currentFeatures[i] + 1 << (i == currentFeatures.size() - 1 ? "" : ", ");
    }
    std::cout << "} with accuracy " << globalBestAcc * 100 << "%\n";
    std::cout << "Beginning backward elimination.\n";

    for (size_t k = 0; k < numFeatures -1; ++k) { // Iterate numFeatures-1 times at most (cannot remove last feature)
        if (currentFeatures.size() <= 1) {
             std::cout << "\nOnly one feature remaining. Halting backward elimination.\n";
            break;
        }
        std::cout << "\nOn level " << k + 1 << " of the search tree\n";
        std::cout << "Current feature set to evaluate for removal: {";
        for (size_t i = 0; i < currentFeatures.size(); ++i) {
            std::cout << currentFeatures[i] + 1 << (i == currentFeatures.size() - 1 ? "" : ", ");
        }
        std::cout << "}\n";


        double bestLocalAcc = -1.0;
        int featureToRemoveThisLevel = -1;

        for (int feature_to_potentially_remove : currentFeatures) {
            std::vector<int> trialFeatures = currentFeatures;
            trialFeatures.erase(std::remove(trialFeatures.begin(), trialFeatures.end(), feature_to_potentially_remove), trialFeatures.end());

            if (trialFeatures.empty()) continue; // Should not happen if currentFeatures.size() > 1

            std::vector<std::vector<double> > projectedX;
            for (const auto& row : X) {
                std::vector<double> proj;
                for (int idx : trialFeatures) {
                    proj.push_back(row[idx]);
                }
                projectedX.push_back(proj);
            }
            double acc = KNNUtils::nnLeaveOneOutCV(projectedX, y);
            std::cout << "    Considering removing feature " << feature_to_potentially_remove + 1 << ". Remaining set {";
            for(size_t i = 0; i < trialFeatures.size(); ++i) {
                std::cout << trialFeatures[i] + 1 << (i == trialFeatures.size() - 1 ? "" : ", ");
            }
            std::cout << "} accuracy is " << acc * 100 << "%\n";

            if (acc >= bestLocalAcc) { // Note: >= to prefer smaller feature sets with same accuracy
                bestLocalAcc = acc;
                featureToRemoveThisLevel = feature_to_potentially_remove;
            }
        }

        if (featureToRemoveThisLevel != -1) {
            currentFeatures.erase(std::remove(currentFeatures.begin(), currentFeatures.end(), featureToRemoveThisLevel), currentFeatures.end());
             std::sort(currentFeatures.begin(), currentFeatures.end());


            std::cout << "\nOn level " << k + 1 << ", removed feature " << featureToRemoveThisLevel + 1 << ". Accuracy with remaining features: " << bestLocalAcc * 100 << "%\n";
            std::cout << "Current best feature set: {";
            for(size_t i = 0; i < currentFeatures.size(); ++i) {
                std::cout << currentFeatures[i] + 1 << (i == currentFeatures.size() - 1 ? "" : ", ");
            }
            std::cout << "} with accuracy " << bestLocalAcc * 100 << "%\n";


            if (bestLocalAcc >= globalBestAcc) { // Prefer smaller set if accuracy is same or better
                globalBestAcc = bestLocalAcc;
                bestFeaturesOverall = currentFeatures;
            } else {
                 std::cout << "(Warning, accuracy has decreased. Global best is still better.)\n";
            }
        } else {
            std::cout << "\nCould not determine a feature to remove or no feature removal improved/maintained accuracy. Halting backward elimination.\n";
            break;
        }
    }

    std::cout << "\nFinished backward elimination!! The best feature subset is: {";
    for (size_t i = 0; i < bestFeaturesOverall.size(); ++i) {
        std::cout << bestFeaturesOverall[i] + 1 << (i == bestFeaturesOverall.size() - 1 ? "" : ", ");
    }
    std::cout << "}, which has an accuracy of " << globalBestAcc * 100 << "%\n";
}