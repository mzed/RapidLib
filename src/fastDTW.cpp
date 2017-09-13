//
//  fastDTW.cpp
//  RapidAPI
//
//  Created by mzed on 07/09/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#include "fastDTW.h"
#include "dtw.h"
#include "timeSeries.h"

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
    std::vector<std::vector<double>> shrunkenX = downsample(seriesX, resolution);
    std::vector<std::vector<double>> shrunkenY = downsample(seriesY, resolution);
    
    
    ///make it fast here
    
    //SearchWindow window = new ExpandedResWindow(tsI, tsJ, shrunkI, shrunkJ, FastDTW.getWarpPathBetween(shrunkI, shrunkJ, searchRadius, distFn), searchRadius);
    
    return dtw.getCost(seriesX, seriesY);
};

std::vector<std::vector<double> > fastDTW::downsample(const std::vector<std::vector<double>> &series, double resolution  = 2.0) {
    std::vector<std::vector<double> > shrunkenSeries = series;
    
    
    return shrunkenSeries;
}