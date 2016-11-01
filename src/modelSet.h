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
    
protected:
    std::vector<baseModel*> myModelSet;
    int numInputs;
    int numOutputs;
    bool created;

#ifndef EMSCRIPTEN //The javascript code will do its own JSON parsing
public:
    std::string getJSON();
    void writeJSON(std::string filepath);
    bool putJSON(std::string jsonMessage);
    bool readJSON(std::string filepath);
    
private:
    Json::Value parse2json();
    void json2modelSet(Json::Value);

#endif
};

#endif
