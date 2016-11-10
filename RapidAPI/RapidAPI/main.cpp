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
    std::string filepath = "/var/tmp/modelSetDescription.json";
    myNN.writeJSON(filepath);

    
    regression myNNfromString;
    myNNfromString.putJSON(myNN.getJSON());
    
    regression myNNfromFile;
    myNNfromFile.readJSON(filepath);
    std::vector<double> inputVec = { 2.0, 44.2 };
    std::cout << "before: " << myNN.process(inputVec)[0] << std::endl;
    std::cout << "from string: " << myNNfromString.process(inputVec)[0] << std::endl;
    std::cout << "from file: " << myNNfromFile.process(inputVec)[0] << std::endl;
    
    ///////////////////////////
    
    myKnn.train(trainingSet);
    std::cout << myKnn.getJSON() << std::endl;
    std::string filepath2 = "/var/tmp/modelSetDescription_knn.json";
    myKnn.writeJSON(filepath2);
    
    classification myKnnFromString;
    myKnnFromString.putJSON(myKnn.getJSON());
    
    classification myKnnfromFile;
    myKnnfromFile.readJSON(filepath2);
    
    std::cout << "knn before: " << myKnn.process(inputVec)[0] << std::endl;
    std::cout << "knn from string: " << myKnnFromString.process(inputVec)[0] << std::endl;
    std::cout << "knn from file: " << myKnnfromFile.process(inputVec)[0] << std::endl;
    
    return 0;
}
