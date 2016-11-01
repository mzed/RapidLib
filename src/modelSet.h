#ifndef modelSet_h
#define modelSet_h

#include <vector>
#include "baseModel.h"
#include "neuralNetwork.h"
#include "knnClassification.h"
#ifndef EMSCRIPTEN
#include "json.h"
#endif

/** This class holds a set of models with the same or different algorithms. */

class modelSet {
public:
    modelSet();
    ~modelSet();
    bool train(std::vector<trainingExample> trainingSet);
    bool initialize();
    std::vector<double> process(std::vector<double> inputVector);
    
    std::string getJSON();
    void writeJSON(std::string filepath);
    bool readJSON(std::string filepath);

protected:
    std::vector<baseModel*> myModelSet;
    int numInputs;
    int numOutputs;
    bool created;
    
private:
    Json::Value parse2json();
};

#endif
