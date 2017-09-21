//
//  seriesClassification.cpp
//
//  Created by Michael Zbyszynski on 08/06/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#include <vector>
#include "seriesClassification.h"
#ifdef EMSCRIPTEN
#include "emscripten/seriesClassificationEmbindings.h"
#endif

seriesClassification::seriesClassification() {};

seriesClassification::~seriesClassification() {};

bool seriesClassification::train(const std::vector<trainingSeries> &seriesSet) {
    reset();
    bool trained = true;
    allTrainingSeries = seriesSet;
    minLength = maxLength = int(allTrainingSeries[0].input.size());
    for (int i = 0; i < allTrainingSeries.size(); ++i) {
        if (allTrainingSeries[i].input.size() < minLength) {
            minLength = int(allTrainingSeries[i].input.size());
        }
        if (allTrainingSeries[i].input.size() > maxLength) {
            maxLength = int(allTrainingSeries[i].input.size());
        }
    }

    //TODO: calculate some size statistics here?
    //min length per label
    //max length per label
    
    return trained;
};

void seriesClassification::reset() {
    allCosts.clear();
    allTrainingSeries.clear();
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

//
//std::vector<double> seriesClassification::getCosts(const std::vector<trainingExample> &trainingSet) {
//    run(trainingSet);
//    return allCosts;
//}