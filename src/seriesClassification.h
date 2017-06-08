//
//  seriesClassification.hp
//  RapidAPI
//
//  Created by mzed on 08/06/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#ifndef seriesClassification_hpp
#define seriesClassification_hpp

#include <vector>
#include "dtw.h"

class seriesClassification {
    
public:
    seriesClassification();
    ~seriesClassification();
    
    bool addSeries(std::vector<std::vector<double>> newSeries);
    void clear();
    
    int process(std::vector<std::vector<double>> inputSeries);
    
private:
    std::vector<dtw> dtwClassifiers;

};

#endif
