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

/* Just returns the cost, doesn't calculate the path */
double dtw::getCost(const std::vector<std::vector<double> > &seriesX, const std::vector<std::vector<double> > &seriesY) {
    if (seriesX.size() < seriesY.size()) {
        return getCost(seriesY, seriesX);
    }
    
    costMatrix.clear();
    for (int i = 0; i < seriesX.size(); ++i) {
        std::vector<double> tempVector;
        for (int j = 0; j < seriesY.size(); ++j) {
            tempVector.push_back(0);
        }
        costMatrix.push_back(tempVector);
    }
    int maxX = int(seriesX.size()) - 1;
    int maxY = int(seriesY.size()) - 1;
    
    //Calculate values for the first column
    costMatrix[0][0] = distanceFunction(seriesX[0], seriesY[0]);
    for (int j = 1; j <= maxY; ++j) {
        costMatrix[0][j] = costMatrix[0][j - 1] + distanceFunction(seriesX[0], seriesY[j]);
    }
    
    for (int i = 1; i <= maxX; ++i) {
        //Bottom row of current column
        costMatrix[i][0] = costMatrix[i - 1][0] + distanceFunction(seriesX[i], seriesY[0]);
        for (int j = 1; j <= maxY; ++j) {
            double minGlobalCost = fmin(costMatrix[i-1][j-1], costMatrix[i][j-1]);
            costMatrix[i][j] = minGlobalCost + distanceFunction(seriesX[i], seriesY[j]);
        }
    }
    cost = costMatrix[maxX][maxY];
    return cost;
};

/* calculates both the cost and the warp path*/
void dtw::dynamicTimeWarp(const std::vector<std::vector<double> > &seriesX, const std::vector<std::vector<double> > &seriesY) {
    
    //calculate cost matrix
    cost = getCost(seriesX, seriesY);
    
    //find path
    int i = int(seriesX.size()) - 1;
    int j = int(seriesY.size()) - 1;
    warpPath.add(i, j);
    
    while ((i > 0) || (j > 0)) {
        double diagonalCost = std::numeric_limits<double>::infinity();
        double leftCost = std::numeric_limits<double>::infinity();
        double downCost = std::numeric_limits<double>::infinity();
        if ((i > 0) && (j > 0)) {
            diagonalCost = costMatrix[i - 1][j - 1];
        }
        if (i > 0) {
            leftCost = costMatrix[i - 1][j];
        }
        if (j > 0) {
            downCost = costMatrix[i][j - 1];
        }
        if ((diagonalCost <= leftCost) && (diagonalCost <= downCost)) {
            i--;
            j--;
        } else if ((leftCost < diagonalCost) && (leftCost < downCost)){
            i--;
        } else if ((downCost < diagonalCost) && (downCost < leftCost)) {
            j--;
        } else if (i <= j) {
            j--;
        } else {
            i--;
        }
        warpPath.add(i, j);
    }
}

/*
 void dtw::constrainedDTW(const std::vector<std::vector<double> > &seriesX, const std::vector<std::vector<double> > &seriesY, window) {
 
 }
 */

warpPath dtw::getPath() {
    return warpPath;
}