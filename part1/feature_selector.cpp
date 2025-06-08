#include "feature_selector.h"
#include "knn_utils.h" 
#include <iostream>
#include <algorithm> 
#include <numeric>   
#include <limits>    

std::vector<std::pair<std::vector<int>, double>> FeatureSelector::forwardSelection(
    const std::vector<std::vector<double> >& X, const std::vector<int>& y) {
    std::vector<std::pair<std::vector<int>, double>> results;
    
    if (X.empty() || X[0].empty()) {
        std::cout << "Input data X is empty or has no features. Aborting forward selection." << std::endl;
        return results;
    }
    
    size_t numFeatures = X[0].size();
    std::vector<int> allFeatures(numFeatures);
    std::iota(allFeatures.begin(), allFeatures.end(), 0);
    std::vector<int> selectedFeatures;

    double globalBestAcc = -1.0;
    std::vector<int> bestFeaturesOverall;

    std::cout << "Beginning forward selection.\n";

    for (size_t k = 0; k < numFeatures; ++k) {
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
            std::vector<int> trialFeatures = selectedFeatures;
            trialFeatures.push_back(featureToConsider);
            std::sort(trialFeatures.begin(), trialFeatures.end());

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
            allFeatures.erase(std::remove(allFeatures.begin(), allFeatures.end(), featureToAddThisLevel), allFeatures.end());

            std::cout << "\nOn level " << k + 1 << ", added feature " << featureToAddThisLevel + 1 << " to current set. Accuracy: " << bestLocalAcc * 100 << "%\n";
            std::cout << "Current best feature set: {";
            for(size_t i = 0; i < selectedFeatures.size(); ++i) {
                std::cout << selectedFeatures[i] + 1 << (i == selectedFeatures.size() - 1 ? "" : ", ");
            }
            std::cout << "} with accuracy " << bestLocalAcc * 100 << "%\n";

            // Store the current result
            results.push_back({selectedFeatures, bestLocalAcc});

            if (bestLocalAcc > globalBestAcc) {
                globalBestAcc = bestLocalAcc;
                bestFeaturesOverall = selectedFeatures;
            } else {
                std::cout << "(Warning, accuracy has decreased or stayed the same. Global best is still better.)\n";
            }
        } else {
            std::cout << "\nNo feature improved accuracy at this level. Halting forward selection.\n";
            break;
        }
        if (selectedFeatures.size() == numFeatures) break;
    }

    std::cout << "\nFinished forward selection!! The best feature subset is: {";
    for (size_t i = 0; i < bestFeaturesOverall.size(); ++i) {
        std::cout << bestFeaturesOverall[i] + 1 << (i == bestFeaturesOverall.size() - 1 ? "" : ", ");
    }
    std::cout << "}, which has an accuracy of " << globalBestAcc * 100 << "%\n";

    return results;
}

std::vector<std::pair<std::vector<int>, double>> FeatureSelector::backwardElimination(
    const std::vector<std::vector<double> >& X, const std::vector<int>& y) {
    std::vector<std::pair<std::vector<int>, double>> results;
    
    if (X.empty() || X[0].empty()) {
        std::cout << "Input data X is empty or has no features. Aborting backward elimination." << std::endl;
        return results;
    }
    
    size_t numFeatures = X[0].size();
    std::vector<int> currentFeatures(numFeatures);
    std::iota(currentFeatures.begin(), currentFeatures.end(), 0);

    std::cout << "Calculating initial accuracy with all features.\n";
    double globalBestAcc = KNNUtils::nnLeaveOneOutCV(X, y);
    std::vector<int> bestFeaturesOverall = currentFeatures;

    // Store initial result
    results.push_back({currentFeatures, globalBestAcc});

    std::cout << "Initial feature set: {";
    for(size_t i = 0; i < currentFeatures.size(); ++i) {
        std::cout << currentFeatures[i] + 1 << (i == currentFeatures.size() - 1 ? "" : ", ");
    }
    std::cout << "} with accuracy " << globalBestAcc * 100 << "%\n";
    std::cout << "Beginning backward elimination.\n";

    for (size_t k = 0; k < numFeatures - 1; ++k) {
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

            if (trialFeatures.empty()) continue;

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

            if (acc >= bestLocalAcc) {
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

            // Store the current result
            results.push_back({currentFeatures, bestLocalAcc});

            if (bestLocalAcc >= globalBestAcc) {
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

    return results;
}