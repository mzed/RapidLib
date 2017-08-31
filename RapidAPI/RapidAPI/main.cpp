#include <vector>
#include <iostream>
#include <cassert>
#include <random>
#include <algorithm>
#include "regression.h"
#include "classification.h"
#include "seriesClassification.h"
#include "json.h"

int main(int argc, const char * argv[]) {
    
    //simple test
    regression myNN2;
    
    std::vector<trainingExample> trainingSet1;
    trainingExample tempExample1;
    tempExample1.input = { 1.0, 1.0, 1.0 };
    tempExample1.output = { 10.0 };
    trainingSet1.push_back(tempExample1);
    tempExample1.input = { 2.0, 2.0, 2.0 };
    tempExample1.output = { 1.3 };
    trainingSet1.push_back(tempExample1);
    myNN2.setNumHiddenLayers(2);
    assert(myNN2.getNumHiddenLayers()[0] == 2);

    myNN2.setEpochs(50000);
    myNN2.train(trainingSet1);
    
    std::vector<double> inputVec1 = { 2.0, 2.0, 2.0 };
    std::cout << "multilayer " << myNN2.run(inputVec1)[0] << std::endl;
    
    return 0;
}
