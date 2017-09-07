//
//  seriesClassification.h
//  RapidAPI
//
//  Created by Michael Zbyszynski on 08/06/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#ifndef seriesClassification_hpp
#define seriesClassification_hpp

#include <vector>
#include <string>
#include "dtw.h"
#include "trainingExample.h"

class seriesClassification {
    
public:
    seriesClassification();
    ~seriesClassification();
    
    bool trainLabel(const std::vector<trainingSeries> &seriesSet);
    
    void reset();
    
    std::string runLabel(const std::vector<std::vector<double>> &inputSeries);
    
    
private:
    std::vector<trainingSeries> allTrainingSeries;
    std::vector<double> allCosts;
    
};

#endif
