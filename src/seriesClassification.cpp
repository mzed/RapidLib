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

bool seriesClassification::addSeries(const std::vector<std::vector<double>> &newSeries) {
    dtw newDTW;
    newDTW.setSeries(newSeries);
    dtwClassifiers.push_back(newDTW);
    return true;
}

bool seriesClassification::addSeries(const std::vector<trainingExample> &trainingSet) {
    std::vector<std::vector<double>> newSeries;
    for (int i = 0; i < trainingSet.size(); ++i) {
        newSeries.push_back(trainingSet[i].input);
    }
    return addSeries(newSeries);
};

///////////////////////////////////////////////// Training
//TODO: Refactor these

bool seriesClassification::train(const std::vector<std::vector<std::vector<double> > > &vectorSet) {
    bool trained = true;
    reset();
    for (int i = 0; i < vectorSet.size(); ++i) {
        if (!addSeries(vectorSet[i])) {
            trained = false;
        };
    }
    return trained;
}

bool seriesClassification::train(const std::vector<std::vector<trainingExample> > &exampleSet) {
    bool trained = true;
    reset();
    for (int i = 0; i < exampleSet.size(); ++i) {
        if (!addSeries(exampleSet[i])) {
            trained = false;
        };
    }
    return trained;
}

bool seriesClassification::trainLabel(const std::vector<trainingSeries> &seriesSet) {
    bool trained = true;
    reset();
    for (int i = 0; i < seriesSet.size(); ++i) {
        if(!addSeries(seriesSet[i].input) ) {
            trained = false;
        }
        labels.push_back(seriesSet[i].label);
    }
    return trained;
};

/////////////////////////////////////////////////

void seriesClassification::reset() {
    labels.clear();
    dtwClassifiers.clear();
}

int seriesClassification::run(const std::vector<std::vector<double>> &inputSeries) {
    //TODO: check vector sizes and reject bad data
    int closestSeries = 0;
    allCosts.clear();
    double lowestCost = dtwClassifiers[0].run(inputSeries);
    allCosts.push_back(lowestCost);
    for (int i = 1; i < dtwClassifiers.size(); ++i) {
        double currentCost = dtwClassifiers[i].run(inputSeries);
        allCosts.push_back(currentCost);
        if (currentCost < lowestCost) {
            lowestCost = currentCost;
            closestSeries = i;
        }
    }
    return closestSeries;
};

int seriesClassification::run(const std::vector<trainingExample> &trainingSet) {
    std::vector<std::vector<double>> newSeries;
    for (int i = 0; i < trainingSet.size(); ++i) {
        newSeries.push_back(trainingSet[i].input);
    }
    return run(newSeries);
};

std::string seriesClassification::runLabel(const std::vector<std::vector<double>> &inputSeries) {
    return labels[run(inputSeries)];
};

std::vector<double> seriesClassification::getCosts() {
    return allCosts;
}

std::vector<double> seriesClassification::getCosts(const std::vector<trainingExample> &trainingSet) {
    run(trainingSet);
    return allCosts;
}