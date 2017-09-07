//
//  dtw.h
//  RapidAPI
//
//  Created by mzed on 07/06/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#ifndef dtw_h
#define dtw_h

#include <vector>

class dtw {
    
public:
    dtw();
    ~dtw();
    
    //void setSeries(std::vector<std::vector<double>> newSeries);
    double getCost(const std::vector<std::vector<double>> &seriesX, const std::vector<std::vector<double > > &seriesY);
    //void reset();
    
private:
    //std::vector<std::vector<double>> storedSeries;
    //int numFeatures;
    inline double distanceFunction(const std::vector<double> &pointX, const std::vector<double> &point);

};

#endif /* dtw_h */
