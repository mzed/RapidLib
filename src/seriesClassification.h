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

template<typename T>
class seriesClassification {
    
public:
    seriesClassification();
    ~seriesClassification();
    
    bool train(const std::vector<trainingSeries<T> > &seriesSet);
    void reset();
    std::string run(const std::vector<std::vector<T>> &inputSeries);
    std::vector<T> getCosts();
    
    int getMinLength();
    int getMinLength(std::string label);
    int getMaxLength();
    int getMaxLength(std::string label);
    
    template<typename TT>
    struct minMax {
        TT min;
        TT max;
    };
    
    minMax<T> calculateCosts(std::string label);
    minMax<T> calculateCosts(std::string label1, std::string label2);
    
private:
    std::vector<trainingSeries<T > > allTrainingSeries;
    std::vector<T> allCosts;
    
    int maxLength;
    int minLength;
    std::map<std::string, minMax<int> > lengthsPerLabel;
};

#endif
