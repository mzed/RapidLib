//
//  dtw.cpp
//  RapidAPI
//
//  Created by mzed on 07/06/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#include <vector>
#include <cmath>
#include <cassert>
#include "dtw.h"

dtw::dtw() {};

dtw::~dtw() {};

//void dtw::setSeries(std::vector<std::vector<double>> newSeries) {
//    storedSeries = newSeries;
//    numFeatures = int(storedSeries[0].size());
//};

inline double dtw::distanceFunction(const std::vector<double> &x, const std::vector<double> &y) {
    assert(x.size() == y.size());
    double euclidianDistance = 0;
    for(int j = 0; j < x.size() ; ++j){
        euclidianDistance = euclidianDistance + pow((x[j] - y[j]), 2);
    }
    euclidianDistance = sqrt(euclidianDistance);
    return euclidianDistance;
};

double dtw::getCost(const std::vector<std::vector<double> > &seriesX, const std::vector<std::vector<double> > &seriesY) {
    if (seriesX.size() < seriesY.size()) {
        return getCost(seriesY, seriesX);
    }
    
    std::vector<std::vector<double> > costMatrix(seriesX.size(), std::vector<double>(seriesY.size(), 0));
    int maxInput = int(seriesX.size()) - 1;
    int maxStored = int(seriesY.size()) - 1;
    
    //Calculate values for the first column
    costMatrix[0][0] = distanceFunction(seriesX[0], seriesY[0]);
    for (int j = 1; j <= maxStored; ++j) {
        costMatrix[0][j] = costMatrix[0][j - 1] + distanceFunction(seriesX[0], seriesY[j]);
    }
    
    for (int i = 1; i <= maxInput; ++i) {
        //Bottom row of current column
        costMatrix[i][0] = costMatrix[i - 1][0] + distanceFunction(seriesX[i], seriesY[0]);
        for (int j = 1; j <= maxStored; ++j) {
            double minGlobalCost = fmin(costMatrix[i-1][j-1], costMatrix[i][j-1]);
            costMatrix[i][j] = minGlobalCost + distanceFunction(seriesX[i], seriesY[j]);
        }
    }
    double minimumCost = costMatrix[maxInput][maxStored];
    return minimumCost;
};