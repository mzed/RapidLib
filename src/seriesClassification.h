//
//  seriesClassification.h
//  RapidLib
//
//  Created by Michael Zbyszynski on 08/06/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#ifndef seriesClassification_hpp
#define seriesClassification_hpp

#include <vector>
#include <string>
#include <map>
#include "fastDTW.h"
#include "trainingExample.h"

class seriesClassification {
    
public:
    seriesClassification();
    ~seriesClassification();
    
    bool train(const std::vector<trainingSeries<double> > &seriesSet);
    void reset();
    std::string run(const std::vector<std::vector<double>> &inputSeries);
    std::vector<double> getCosts();
    
    int getMinLength();
    int getMinLength(std::string label);
    int getMaxLength();
    int getMaxLength(std::string label);
    
    template<typename T>
    struct minMax {
        T min;
        T max;
    };
    
    minMax<double> calculateCosts(std::string label);
    minMax<double> calculateCosts(std::string label1, std::string label2);
    
private:
    std::vector<trainingSeries<double > > allTrainingSeries;
    std::vector<double> allCosts;
    
    int maxLength;
    int minLength;
    std::map<std::string, minMax<int> > lengthsPerLabel;
};

#endif
