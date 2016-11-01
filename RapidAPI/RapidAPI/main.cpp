#include <vector>
#include <iostream>
#include "regression.h"
#include "classification.h"
#include "json.h"

int main(int argc, const char * argv[]) {
    // insert code here...
    
    regression myNN;
    classification myKnn;
    
    std::vector<trainingExample> trainingSet;
    trainingExample tempExample;
    tempExample.input = { 0.2, 0.7 };
    tempExample.output = { 3.0 };
    trainingSet.push_back(tempExample);

    tempExample.input = { 2.0, 44.2 };
    tempExample.output = { 20.14 };
    trainingSet.push_back(tempExample);
    
    myNN.train(trainingSet);
    myNN.writeJSON();

    
    regression myNNfromFile;
    
    myNNfromFile.readJSON();
    std::vector<double> inputVec = { 2.0, 44.2 };
    std::cout << myNNfromFile.process(inputVec).size() << std::endl;
    
    return 0;
}
