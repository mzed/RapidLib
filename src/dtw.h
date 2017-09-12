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
    
    double getCost(const std::vector<std::vector<double>> &seriesX, const std::vector<std::vector<double > > &seriesY);
    
private:
    inline double distanceFunction(const std::vector<double> &pointX, const std::vector<double> &point);

};

#endif /* dtw_h */
