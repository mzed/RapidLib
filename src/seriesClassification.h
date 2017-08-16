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
    
    bool addSeries(const std::vector<std::vector<double>> &newSeries);
    bool addSeries(const std::vector<trainingExample> &trainingSet);
    
    bool train(const std::vector<std::vector<std::vector<double> > > &vectorSet);
    bool train(const std::vector<std::vector<trainingExample>> &exampleSet);
    
    bool trainLabel(const std::vector<trainingSeries> &seriesSet);
    
    void reset();
    
    int run(const std::vector<std::vector<double>> &inputSeries);
    int run(const std::vector<trainingExample> &inputSet);
    std::string runLabel(const std::vector<std::vector<double>> &inputSeries);
    
    std::vector<double> getCosts();
    std::vector<double> getCosts(const std::vector<trainingExample> &inputSet);
    
private:
    std::vector<std::string> labels;
    std::vector<dtw> dtwClassifiers;
    std::vector<double> allCosts;
    
};

#endif
