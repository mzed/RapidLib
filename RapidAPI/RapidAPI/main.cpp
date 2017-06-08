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
    
    regression myNN;
    classification myKnn;
    classification mySVM(classification::svm);
    
    std::vector<trainingExample> trainingSet;
    trainingExample tempExample;
    tempExample.input = { 0.2, 0.7 };
    tempExample.output = { 3.0 };
    trainingSet.push_back(tempExample);
    
    tempExample.input = { 2.0, 44.2 };
    tempExample.output = { 20.14 };
    trainingSet.push_back(tempExample);
    
    myNN.train(trainingSet);
//    std::cout << myNN.getJSON() << std::endl;
    std::string filepath = "/var/tmp/modelSetDescription.json";
    myNN.writeJSON(filepath);
    
    
    regression myNNfromString;
    myNNfromString.putJSON(myNN.getJSON());
    
    regression myNNfromFile;
    myNNfromFile.readJSON(filepath);
    std::vector<double> inputVec = { 2.0, 44.2 };
    
    std::cout << "before: " << myNN.process(inputVec)[0] << std::endl;
    std::cout << "from string: " << myNNfromString.process(inputVec)[0] << std::endl;
 //   std::cout << myNNfromString.getJSON() << std::endl;
    std::cout << "from file: " << myNNfromFile.process(inputVec)[0] << std::endl;
    
    assert(myNN.process(inputVec)[0] == myNNfromString.process(inputVec)[0]);
    assert(myNN.process(inputVec)[0] == myNNfromFile.process(inputVec)[0]);
    
    ///////////////////////////
    
    myKnn.train(trainingSet);
    mySVM.train(trainingSet);
    
 //   std::cout << myKnn.getJSON() << std::endl;
    std::string filepath2 = "/var/tmp/modelSetDescription_knn.json";
    myKnn.writeJSON(filepath2);
    
    classification myKnnFromString(classification::knn);
    myKnnFromString.putJSON(myKnn.getJSON());
    
    classification myKnnFromFile;
    myKnnFromFile.readJSON(filepath2);
    
    std::cout << "knn before: " << myKnn.process(inputVec)[0] << std::endl;
    std::cout << "svm: " << mySVM.process(inputVec)[0] << std::endl;
    std::cout << "knn from string: " << myKnnFromString.process(inputVec)[0] << std::endl;
    std::cout << "knn from file: " << myKnnFromFile.process(inputVec)[0] << std::endl;
    
    assert(myKnn.process(inputVec)[0] == myKnnFromString.process(inputVec)[0]);
    assert(myKnn.process(inputVec)[0] == myKnnFromFile.process(inputVec)[0]);
    
    assert(myKnn.getK()[0] == 1);
    myKnn.setK(0, 2);
    assert(myKnn.getK()[0] == 2);

    regression bigVector;
    std::vector<trainingExample> trainingSet2;
    trainingExample tempExample2;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.5,0.5);
    int vecLength = 64;
    for (int j = 0; j < vecLength; ++j) {
        tempExample2.input.clear();
        tempExample2.output.clear();
        for (int i = 0; i < vecLength; ++i) {
            tempExample2.input.push_back(distribution(generator));
        }
        tempExample2.output = { distribution(generator) };
        trainingSet2.push_back(tempExample2);
    }
    bigVector.train(trainingSet2);
    std::vector<double> inputVec2;
    for (int i=0; i < vecLength; ++i) {
        inputVec2.push_back(distribution(generator));
    }
    assert (isfinite(bigVector.process(inputVec2)[0]));
    
    
    /////////

    classification mySVM2(classification::svm);
    
    std::vector<trainingExample> trainingSet3;
    trainingExample tempExample3;
    
    tempExample3.input = { 0., 0. };
    tempExample3.output = { 0. };
    trainingSet3.push_back(tempExample3);
    
    tempExample3.input = { 1., 0. };
    tempExample3.output = { 1. };
    trainingSet3.push_back(tempExample3);
    
    tempExample3.input = { 1., 8. };
    tempExample3.output = { 5. };
    trainingSet3.push_back(tempExample3);
    
    

    /*
    tempExample3.input = { 1., 1. };
    tempExample3.output = { 2. };
    trainingSet3.push_back(tempExample3);
    */
    
    mySVM2.train(trainingSet3);

    std::vector<double>inputVec4 = { 1., 0. };
    std::cout << "svm: " << mySVM2.process(inputVec4)[0] << std::endl;
    
    std::vector<double> inputVec3 = { 0., 0. };
    std::cout << "svm2: " << mySVM2.process(inputVec3)[0] << std::endl;
    

    
    ////////////
    
    seriesClassification myDtw;
    
    std::vector<std::vector<double>> seriesOne;
    seriesOne.push_back( { 1., 5.} );
    seriesOne.push_back( { 2., 4.} );
    seriesOne.push_back( { 3., 3.} );
    seriesOne.push_back( { 4., 2.} );
    seriesOne.push_back( { 5., 1.} );
    myDtw.addSeries(seriesOne);
    
    std::vector<std::vector<double>> seriesTwo;
    seriesTwo.push_back( { 1., 4. } );
    seriesTwo.push_back( { 2., -3. } );
    seriesTwo.push_back( { 1., 5. } );
    seriesTwo.push_back( { -2., 1. } );
    myDtw.addSeries(seriesTwo);
    
    std::cout << "dtw: " << myDtw.process(seriesOne) << std::endl;
    std::cout << "dtw: " << myDtw.process(seriesTwo) << std::endl;
    
    
    seriesClassification myDtw2;
    std::vector<trainingExample> tsOne;
    
    tempExample.input = { 1., 5. };
    tempExample.output = { 0.0 };
    tsOne.push_back(tempExample);
    
    tempExample.input = { 2., 4. };
    tempExample.output = { 0.0 };
    tsOne.push_back(tempExample);
    
    tempExample.input = { 3., 3. };
    tempExample.output = { 0.0 };
    tsOne.push_back(tempExample);
    
    tempExample.input = { 4., 2. };
    tempExample.output = { 0.0 };
    tsOne.push_back(tempExample);
    
    tempExample.input = { 5., 1. };
    tempExample.output = { 0.0 };
    tsOne.push_back(tempExample);
    
    myDtw2.addTrainingSet(tsOne);
    
    std::vector<trainingExample> tsTwo;
    tempExample.input = { 1., 4. };
    tempExample.output = { 0.0 };
    tsTwo.push_back(tempExample);
    
    tempExample.input = { 2., -3. };
    tempExample.output = { 0.0 };
    tsTwo.push_back(tempExample);
    
    tempExample.input = { 1., 5. };
    tempExample.output = { 0.0 };
    tsTwo.push_back(tempExample);
    
    tempExample.input = { -2., 1. };
    tempExample.output = { 0.0 };
    tsTwo.push_back(tempExample);
    
    myDtw2.addTrainingSet(tsTwo);

    
    std::cout << "dtw2: " << myDtw2.processTrainingSet(tsOne) << std::endl;
    std::cout << "dtw2: " << myDtw2.processTrainingSet(tsTwo) << std::endl;
    
    return 0;
}
