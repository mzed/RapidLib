//
//  timeSeries.cpp
//  RapidAPI
//
//  Created by mzed on 12/09/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#include <cassert>
#include "math.h"
#include "timeSeries.h"


timeSeries::timeSeries(const std::vector<std::vector<double > > &inputSeries, int shrunkenSize){
    assert(shrunkenSize > inputSeries.size());
    assert(shrunkenSize > 0);
    
    originalLength = int(inputSeries.size());
    double reducedPointSize = double(inputSeries.size())/(double(shrunkenSize));
    int pointToReadFrom = 0;
    int pointToReadTo;
    
    while (pointToReadFrom < inputSeries.size()) {
        
        //determine end of current range
        pointToReadTo = (int)round(reducedPointSize * (inputSeries.size() + 1 )) - 1; //FIXME: taking the wrong size here?
        int pointsToRead = pointToReadTo - pointToReadFrom + 1;
        
        //double timeSum = 0.0;  //FIXME: do I need this?
        std::vector<double> measurmentSums; //one feature vector long?
        
        //sum all of the values over the range
        for (int point = pointToReadFrom; point <= pointToReadTo; ++point) {
            std::vector<double> currentPoint = inputSeries[point];
            
        }
        /*
        for (int i = 0; i < currentPoint.size(); ++i) {
            measurementSums[i] += currentPoint[i];
            
        }
        
        //Determain average value over range
        
        
        */
    }
    
    
};

