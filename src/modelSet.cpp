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

std::vector<double> json2vector(Json::Value json) {
    std::vector<double> returnVec;
    for (int i = 0; i < json.size(); ++i) {
        returnVec.push_back(json[i].asDouble());
    }
    return returnVec;
}

Json::Value modelSet::parse2json() {
    Json::Value root;
    Json::Value metadata;
    Json::Value modelSet;
    
    metadata["creator"] = "Rapid API C++";
    metadata["version"] = "v0.1.1"; //TODO: This should be a macro someplace
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
    return root;
}

std::string modelSet::getJSON() {
    Json::Value root = parse2json();
    return root.toStyledString();
}

void modelSet::writeJSON(std::string filepath) {
    Json::Value root = parse2json();
    std::ofstream jsonOut;
    jsonOut.open (filepath);
    Json::StyledStreamWriter writer;
    writer.write(jsonOut, root);
    jsonOut.close();
    
}

bool modelSet::readJSON(std::string filepath) {
    Json::Value root;
    std::ifstream file(filepath);
    file >> root;
    numInputs = root["metadata"]["numInputs"].asInt();
    numOutputs = root["metadata"]["numOutputs"].asInt();
    
    for (const Json::Value& model : root["modelSet"]) {
        int modelNumInputs = model["numInputs"].asInt();
        std::vector<int> whichInputs;
        for (int i = 0; i < model["whichInputs"].size(); ++i) { //TODO: factor these
            whichInputs.push_back(model["whichInputs"][i].asDouble());
        }
        int numHiddenLayers = model["numHiddenLayers"].asInt();
        int numHiddenNodes = model["numHiddenNodes"].asInt();
        std::vector<double> weights = json2vector(model["weights"]);
        std::vector<double> wHiddenOutput = json2vector(model["wHiddenOutput"]);
        std::vector<double> inBases = json2vector(model["inBases"]);
        std::vector<double> inRanges = json2vector(model["inRanges"]);
        double outRange = model["outRange"].asDouble();
        double outBase = model["outBase"].asDouble();
        
        myModelSet.push_back(new neuralNetwork(modelNumInputs, whichInputs, numHiddenLayers, numHiddenNodes, weights, wHiddenOutput, inRanges, inBases, outRange, outBase));
    }
    created = true;
    return true; //TODO: check something first
}
#endif