//
//  seriesClassification.cpp
//  RapidAPI
//
//  Created by mzed on 08/06/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#include <vector>
#include "seriesClassification.h"
#ifdef EMSCRIPTEN
#include "emscripten/seriesClassificationEmbindings.h"
#endif


seriesClassification::seriesClassification() {};

seriesClassification::~seriesClassification() {};

bool seriesClassification::addSeries(std::vector<std::vector<double>> newSeries) {
    dtw newDTW;
    newDTW.setSeries(newSeries);
    dtwClassifiers.push_back(newDTW);
    return true;
}

void seriesClassification::clear() {
    dtwClassifiers.clear();
}


int seriesClassification::process(std::vector<std::vector<double>> inputSeries) {
    //TODO: check vector sizes and reject bad data
    int closestSeries = 0;
    double lowestCost = dtwClassifiers[0].process(inputSeries);
    for (int i = 1; i < dtwClassifiers.size(); ++i) {
        double currentCost = dtwClassifiers[i].process(inputSeries);
        if (currentCost < lowestCost) {
            lowestCost = currentCost;
            closestSeries = i;
        }
    }
    
    return closestSeries;
};