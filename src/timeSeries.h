//
//  timeSeries.hpp
//  RapidAPI
//
//  Created by mzed on 12/09/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#ifndef timeSeries_h
#define timeSeries_h

#include <vector>

class timeSeries {
public:
    timeSeries(const std::vector<std::vector<double > > &inputSeries, int shrunkenSize);
    
private:
    int originalLength;
    std::vector<int> aggPointSize;
    
};


#endif /* timeSeries_h*/
