//
//  seriesClassification.cpp
//
//  Created by Michael Zbyszynski on 08/06/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#include <vector>
#include <cassert>
#include "seriesClassification.h"
#ifdef EMSCRIPTEN
#include "emscripten/seriesClassificationEmbindings.h"
#endif

seriesClassification::seriesClassification() {};

seriesClassification::~seriesClassification() {};

bool seriesClassification::train(const std::vector<trainingSeries> &seriesSet) {
    assert(seriesSet.size() > 0);
    reset();
    bool trained = true;
    allTrainingSeries = seriesSet;
    minLength = maxLength = int(allTrainingSeries[0].input.size());
    for (int i = 0; i < allTrainingSeries.size(); ++i) {
        //Global
        int newLength = int(allTrainingSeries[i].input.size());
        if (newLength < minLength) {
            minLength = newLength;
        }
        if (newLength > maxLength) {
            maxLength = newLength;
        }
        //Per Label
        std::map<std::string, lengths>::iterator it = lengthsPerLabel.find(allTrainingSeries[i].label);
        if (it != lengthsPerLabel.end()) {
            int newLength = int(allTrainingSeries[i].input.size());
            if (newLength < it->second.min) {
                it->second.min = newLength;
            }
            if (newLength > it->second.max) {
                it->second.max = newLength;
            }
        } else {
            lengths tempLengths;
            tempLengths.min = tempLengths.max = int(allTrainingSeries[i].input.size());
            lengthsPerLabel[allTrainingSeries[i].label] = tempLengths;
        }
    }
    return trained;
};

void seriesClassification::reset() {
    allCosts.clear();
    allTrainingSeries.clear();
    lengthsPerLabel.clear();
    minLength = -1;
    maxLength = -1;
}

std::string seriesClassification::run(const std::vector<std::vector<double>> &inputSeries) {
    fastDTW fastDtw;
    int searchRadius = 1; //TODO: Define this properly, elsewhere?
    int closestSeries = 0;
    allCosts.clear();
    double lowestCost = fastDtw.getCost(inputSeries, allTrainingSeries[0].input, searchRadius);
    allCosts.push_back(lowestCost);
    
    for (int i = 1; i < allTrainingSeries.size(); ++i) {
        double currentCost = fastDtw.getCost(inputSeries, allTrainingSeries[i].input, searchRadius);
        allCosts.push_back(currentCost);
        if (currentCost < lowestCost) {
            lowestCost = currentCost;
            closestSeries = i;
        }
    }
    return allTrainingSeries[closestSeries].label;
};


std::vector<double> seriesClassification::getCosts() {
    return allCosts;
}

int seriesClassification::getMinLength() {
    return minLength;
}

int seriesClassification::getMinLength(std::string label) {
    int labelMinLength = -1;
    std::map<std::string, lengths>::iterator it = lengthsPerLabel.find(label);
    if (it != lengthsPerLabel.end()) {
        labelMinLength = it->second.min;
    }
    return labelMinLength;
}

int seriesClassification::getMaxLength() {
    return maxLength;
}

int seriesClassification::getMaxLength(std::string label) {
    int labelMaxLength = -1;
    std::map<std::string, lengths>::iterator it = lengthsPerLabel.find(label);
    if (it != lengthsPerLabel.end()) {
        labelMaxLength = it->second.max;
    }
    return labelMaxLength;
}

//
//std::vector<double> seriesClassification::getCosts(const std::vector<trainingExample> &trainingSet) {
//    run(trainingSet);
//    return allCosts;
//}