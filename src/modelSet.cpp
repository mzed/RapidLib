#include <iostream>
#include <vector>
#include "modelSet.h"

#ifndef EMSCRIPTEN
#include "json.h"
#endif

#ifdef EMSCRIPTEN
#include "modelSetEmbindings.h"
#endif

/** No arguments, don't create models yet */

modelSet::modelSet() :
numInputs(0),
numOutputs(0),
created(false)
{
};

modelSet::~modelSet() {
    for (std::vector<baseModel*>::iterator i = myModelSet.begin(); i != myModelSet.end(); ++i) {
        delete *i;
    }
};

bool modelSet::train(std::vector<trainingExample> training_set) {
    for ( auto example : training_set) {
        if (example.input.size() != numInputs) {
            return false;
        }
        if (example.output.size() != numOutputs) {
            return false;
        }
    }
    for (int i = 0; i < myModelSet.size(); ++i) {
        std::vector<trainingExample> modelTrainingSet; //just one output
        for (auto example : training_set) {
            std::vector<double> tempDouble;
            for (int j = 0; j < numInputs; ++j) {
                tempDouble.push_back(example.input[j]);
            }
            trainingExample tempObj = {tempDouble, std::vector<double> {example.output[i]}};
            modelTrainingSet.push_back(tempObj);
        }
        myModelSet[i]->train(modelTrainingSet);
    }
    created = true;
    return true;
}

bool modelSet::initialize() {
    for (std::vector<baseModel*>::iterator i = myModelSet.begin(); i != myModelSet.end(); ++i) {
        delete *i;
    }
    numInputs = 0;
    numOutputs = 0;
    created = false;
    return true;
}

std::vector<double> modelSet::process(std::vector<double> inputVector) {
    std::vector<double> returnVector;
    if (created && inputVector.size() == numInputs) {
        for (auto model : myModelSet) {
            returnVector.push_back(model->process(inputVector));
        }
    } else {
        returnVector.push_back(0);
    }
    return returnVector;
}

#ifndef EMSCRIPTEN
void modelSet::writeJSON() {
    Json::Value root;
    Json::Value metadata;
    Json::Value modelSet;
    
    metadata["creator"] = "Rapid API C++";
    metadata["version"] = "the best one";
    metadata["numInputs"] = numInputs;
    metadata["numOutputs"] = numOutputs;
    root["metadata"] = metadata;

    for (auto model : myModelSet) {
        Json::Value jsonModelDescription;
        jsonModelDescription["modelType"] = "Neural Network"; //FIXME: check type
        jsonModelDescription["numInputs"] = model->getNumInputs();
        Json::Value jsonWhichInputs;
        std::vector<int> whichInputs = model->getWhichInputs();
        for (int i = 0; i < whichInputs.size(); ++i) {
            jsonWhichInputs.append(whichInputs[i]);
        }
        jsonModelDescription["whichInputs"] = jsonWhichInputs;
        //FIXME: needs to work with classifiers, too
        neuralNetwork *nnModel = dynamic_cast<neuralNetwork*>(model);
        jsonModelDescription["numHiddenLayers"];
        jsonModelDescription["numHiddenNodes"];
        jsonModelDescription["numHiddenOutputs"] = 1;
        Json::Value jsonInRanges;
        Json::Value jsonInBases;
        std::vector<double> inRanges = nnModel->getInRanges();
        std::vector<double> inBases = nnModel->getInBases();
        for (int i = 0; i < inRanges.size(); ++i) {
            jsonInRanges.append(inRanges[i]);
            jsonInBases.append(inBases[1]);
        }
        jsonModelDescription["inRanges"] = jsonInRanges;
        jsonModelDescription["inBases"] = jsonInBases;
        jsonModelDescription["outRange"] = nnModel->getOutRange();
        jsonModelDescription["outBase"] = nnModel->getOutBase();
        Json::Value nodes;
        jsonModelDescription["nodes"] = nodes;
        
        modelSet.append(jsonModelDescription);
    }
    root["modelSet"] = modelSet;
    std::cout << root << std::endl;
}
#endif