//
//  seriesClassification.cpp
//  RapidLib
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

#define SEARCH_RADIUS 1

template<typename T>
seriesClassification<T>::seriesClassification() {};

template<typename T>
seriesClassification<T>::~seriesClassification() {};

template<typename T>
bool seriesClassification<T>::train(const std::vector<trainingSeries<T> > &seriesSet) {
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
        typename std::map<std::string, minMax<int> >::iterator it = lengthsPerLabel.find(allTrainingSeries[i].label);
        if (it != lengthsPerLabel.end()) {
            int newLength = int(allTrainingSeries[i].input.size());
            if (newLength < it->second.min) {
                it->second.min = newLength;
            }
            if (newLength > it->second.max) {
                it->second.max = newLength;
            }
        } else {
            minMax<int> tempLengths;
            tempLengths.min = tempLengths.max = int(allTrainingSeries[i].input.size());
            lengthsPerLabel[allTrainingSeries[i].label] = tempLengths;
        }
    }
    return trained;
};

template<typename T>
void seriesClassification<T>::reset() {
    allCosts.clear();
    allTrainingSeries.clear();
    lengthsPerLabel.clear();
    minLength = -1;
    maxLength = -1;
}

template<typename T>
std::string seriesClassification<T>::run(const std::vector<std::vector<T>> &inputSeries) {
    int closestSeries = 0;
    allCosts.clear();
    T lowestCost = fastDTW<T>::getCost(inputSeries, allTrainingSeries[0].input, SEARCH_RADIUS);
    allCosts.push_back(lowestCost);
    
    for (int i = 1; i < allTrainingSeries.size(); ++i) {
        T currentCost = fastDTW<T>::getCost(inputSeries, allTrainingSeries[i].input, SEARCH_RADIUS);
        allCosts.push_back(currentCost);
        if (currentCost < lowestCost) {
            lowestCost = currentCost;
            closestSeries = i;
        }
    }
    return allTrainingSeries[closestSeries].label;
};


template<typename T>
std::vector<T> seriesClassification<T>::getCosts() {
    return allCosts;
}

template<typename T>
int seriesClassification<T>::getMinLength() {
    return minLength;
}

template<typename T>
int seriesClassification<T>::getMinLength(std::string label) {
    int labelMinLength = -1;
    typename std::map<std::string, minMax<int> >::iterator it = lengthsPerLabel.find(label);
    if (it != lengthsPerLabel.end()) {
        labelMinLength = it->second.min;
    }
    return labelMinLength;
}

template<typename T>
int seriesClassification<T>::getMaxLength() {
    return maxLength;
}

template<typename T>
int seriesClassification<T>::getMaxLength(std::string label) {
    int labelMaxLength = -1;
    typename std::map<std::string, minMax<int> >::iterator it = lengthsPerLabel.find(label);
    if (it != lengthsPerLabel.end()) {
        labelMaxLength = it->second.max;
    }
    return labelMaxLength;
}

template<typename T>
seriesClassification<T>::minMax<T> seriesClassification<T>::calculateCosts(std::string label) {
    minMax<T> calculatedMinMax;
    calculatedMinMax.min = std::numeric_limits<T>::max();
    calculatedMinMax.max = std::numeric_limits<T>::min();
    int numSeries = 0;
    
    for (int i = 0; i < (allTrainingSeries.size() - 1); ++i) { //these loops are a little different than the two-label case
        if (allTrainingSeries[i].label == label) {
            for (int j = (i + 1); j < allTrainingSeries.size(); ++j) {
                if (allTrainingSeries[j].label == label) {
                    numSeries++;
                    T currentCost = fastDTW<T>::getCost(allTrainingSeries[i].input, allTrainingSeries[j].input, SEARCH_RADIUS);
                    if (numSeries == 1) {
                        calculatedMinMax.min = calculatedMinMax.max = currentCost; //first match is both min and max
                    } else {
                        if (currentCost < calculatedMinMax.min) {
                            calculatedMinMax.min = currentCost;
                        }
                        if (currentCost > calculatedMinMax.max) {
                            calculatedMinMax.max = currentCost;
                        }
                    }
                }
            }
        }
    }
    if (numSeries == 0) {
        calculatedMinMax.min = calculatedMinMax.max = 0;
    }
    return calculatedMinMax;
}

template<typename T>
seriesClassification<T>::minMax<T> seriesClassification<T>::calculateCosts(std::string label1, std::string label2) {
    minMax<T> calculatedMinMax;
    calculatedMinMax.min = std::numeric_limits<T>::max();
    calculatedMinMax.max = std::numeric_limits<T>::min();
    int numSeries = 0;
    
    for (int i = 0; i < (allTrainingSeries.size()); ++i) {
        if (allTrainingSeries[i].label == label1) {
            for (int j = 0; j < allTrainingSeries.size(); ++j) {
                if (allTrainingSeries[j].label == label2) {
                    numSeries++;
                    T currentCost = fastDTW<T>::getCost(allTrainingSeries[i].input, allTrainingSeries[j].input, SEARCH_RADIUS);
                    if (numSeries == 1) {
                        calculatedMinMax.min = calculatedMinMax.max = currentCost; //first match is both min and max
                    } else {
                        if (currentCost < calculatedMinMax.min) {
                            calculatedMinMax.min = currentCost;
                        }
                        if (currentCost > calculatedMinMax.max) {
                            calculatedMinMax.max = currentCost;
                        }
                    }
                }
            }
        }
    }
    return calculatedMinMax;
}

//explicit instantiation
template class seriesClassification<double>;
template class seriesClassification<float>;


//
//std::vector<T> seriesClassification::getCosts(const std::vector<trainingExample> &trainingSet) {
//    run(trainingSet);
//    return allCosts;
//}