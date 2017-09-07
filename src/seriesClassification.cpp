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

    //TODO: calculate some size statistics here?
    
    return trained;
};

void seriesClassification::reset() {
    allCosts.clear();
    allTrainingSeries.clear();
}

std::string seriesClassification::run(const std::vector<std::vector<double>> &inputSeries) {
    dtw dtw;
    int closestSeries = 0;
    allCosts.clear();
    double lowestCost = dtw.getCost(inputSeries, allTrainingSeries[0].input);
    allCosts.push_back(lowestCost);
    
    for (int i = 1; i < allTrainingSeries.size(); ++i) {
        double currentCost = dtw.getCost(inputSeries, allTrainingSeries[i].input);
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