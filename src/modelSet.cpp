#include <iostream>
#include <fstream>
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

template<typename T>
Json::Value vector2json(T vec) {
    Json::Value toReturn;
    for (int i = 0; i < vec.size(); ++i) {
        toReturn.append(vec[i]);
    }
    return toReturn;
}

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
        jsonModelDescription["whichInputs"] = vector2json(model->getWhichInputs());
        //FIXME: needs to work with classifiers, too
        neuralNetwork *nnModel = dynamic_cast<neuralNetwork*>(model);
        jsonModelDescription["numHiddenLayers"] = nnModel->getNumHiddenLayers();
        jsonModelDescription["numHiddenNodes"] = nnModel->getNumHiddenNodes();
        jsonModelDescription["numHiddenOutputs"] = 1;
        jsonModelDescription["inRanges"] = vector2json(nnModel->getInRanges());
        jsonModelDescription["inBases"] = vector2json(nnModel->getInBases());
        jsonModelDescription["outRange"] = nnModel->getOutRange();
        jsonModelDescription["outBase"] = nnModel->getOutBase();
        jsonModelDescription["weights"]= vector2json(nnModel->getWeights());
        jsonModelDescription["wHiddenOutput"] = vector2json(nnModel->getWHiddenOutput());
        
        modelSet.append(jsonModelDescription);
    }
    root["modelSet"] = modelSet;
    std::cout << root << std::endl;
    
    std::ofstream jo;
    jo.open ("/var/tmp/modelSetDescription.json"); //FIXME: write someplace better, esp for windows
    Json::StyledStreamWriter writer;
    writer.write(jo, root);
    jo.close();
    
}
#endif