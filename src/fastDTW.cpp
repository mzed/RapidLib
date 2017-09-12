//
//  fastDTW.cpp
//  RapidAPI
//
//  Created by mzed on 07/09/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#include "fastDTW.h"
#include "dtw.h"

fastDTW::fastDTW() {};
fastDTW::~fastDTW() {};

double fastDTW::getCost(const std::vector<std::vector<double>> &seriesX, const std::vector<std::vector<double > > &seriesY, int searchRadius){
    dtw dtw;
    searchRadius = (searchRadius < 0) ? 0 : searchRadius;
    //don't bother to do fastDTW if the series are small
    int minSeries = searchRadius + 2;
    if (seriesX.size() <= minSeries || seriesY.size() <= minSeries) {
        return dtw.getCost(seriesX, seriesY);
    }
    
    double resolution = 2.0;
    
    
    ///make it fast here
    
    return dtw.getCost(seriesX, seriesY);
};