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
    std::cout << myNN.getJSON() << std::endl;
    myNN.writeJSON("/var/tmp/modelSetDescription.json");

    
    regression myNNfromFile;
    
    myNNfromFile.readJSON("/var/tmp/modelSetDescription.json");
    std::vector<double> inputVec = { 2.0, 44.2 };
    std::cout << "before: " << myNN.process(inputVec)[0] << std::endl;
    std::cout << "after: " << myNNfromFile.process(inputVec)[0] << std::endl;
    
    return 0;
}
