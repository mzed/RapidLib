#ifndef modelSet_h
#define modelSet_h

#include <vector>
#include "baseModel.h"
#include "neuralNetwork.h"
#include "knnClassification.h"

/** This class holds a set of models with the same or different algorithms. */

class modelSet {
public:
    modelSet();
    ~modelSet();
    bool train(std::vector<trainingExample> trainingSet);
    std::vector<double> process(std::vector<double> inputVector);

protected:
    std::vector<baseModel*> myModelSet;
    int numInputs;
    int numOutputs;
    bool created;
};

#endif
