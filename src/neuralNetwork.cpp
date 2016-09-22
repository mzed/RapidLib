#include <math.h>
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>

#include "neuralNetwork.h"
#include "nnEmbindings.h"

#define DEBUG
//this is the constructor for building a trained model from JSON
neuralNetwork::neuralNetwork(int num_inputs,
                             std::vector<int> which_inputs,
                             int num_hidden_layers,
                             int num_hidden_nodes,
                             std::vector<double> _weights,
                             std::vector<double> w_hidden_output,
                             std::vector<double> in_max,
                             std::vector<double> in_min,
                             double out_max,
                             double out_min
                             )
:
numInputs(num_inputs),
whichInputs(which_inputs),
numHiddenLayers(num_hidden_layers),
numHiddenNodes(num_hidden_nodes),
wHiddenOutput(w_hidden_output),
epoch(0),
learningRate(LEARNING_RATE),
momentum(MOMENTUM),
numEpochs(NUM_EPOCHS),
outputErrorGradient(0)
{
    bool randomize = _weights.size() ? false : true;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.5,0.5);
    //winding up a long vector from javascript
    int count = 0;
    for (int i = 0; i < numHiddenLayers; ++i) {
        std::vector<std::vector<double>> layer;
        for (int j = 0; j < numHiddenNodes; ++j){
            std::vector<double> node;
            for(int k = 0; k <= numInputs; ++k){ //FIXME if numInputs =/= numHiddenNodes
                if (randomize) {
                    node.push_back(distribution(generator));
                } else {
                    node.push_back( _weights[count]);
                }
                count++;
            }
            layer.push_back(node);
        }
        weights.push_back(layer);
    }
    
    if(randomize) {
        for (int i = 0; i <= numHiddenNodes; ++i) {
            wHiddenOutput.push_back(distribution(generator));
        }
    }
    
    for (int i = 0; i < numInputs; ++i) {
        inRanges.push_back((in_max[i] - in_min[i])/ 2);
        inBases.push_back((in_max[i] + in_min[i])/ 2);
    }
    outRange = (out_max - out_min)/ 2;
    outBase = (out_max + out_min)/ 2;
    
    //////////////////////////////////////////trainer
    
    //initialize deltas
    for (int i = 0; i < numHiddenLayers; ++i) {
        std::vector<std::vector<double>> layer;
        for (int j = 0; j < numHiddenNodes; ++j) {
            std::vector<double> node;
            for (int k = 0; k <= numInputs; ++k) { //FIXME if numInputs =/=numHiddenNodes
                node.push_back(0);
            }
            layer.push_back(node);
        }
        deltaWeights.push_back(layer);
    }
    
    for (int i = 0; i <= numHiddenNodes; ++i) {
        deltaHiddenOutput.push_back(0);
    }
    
    //initialize gradients
    for (int i = 0; i <= numHiddenNodes; ++i) {
        hiddenErrorGradients.push_back(0);
    }
}

//this is the constructor for a model that needs to be trained
neuralNetwork::neuralNetwork(int num_inputs,
                             std::vector<int> which_inputs,
                             int num_hidden_layers,
                             int num_hidden_nodes
                             )
:
numInputs(num_inputs),
whichInputs(which_inputs),
numHiddenLayers(num_hidden_layers),
numHiddenNodes(num_hidden_nodes),
epoch(0),
learningRate(LEARNING_RATE),
momentum(MOMENTUM),
numEpochs(NUM_EPOCHS),
outputErrorGradient(0)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.5,0.5);
    
    for (int i = 0; i < numHiddenLayers; ++i) {
        std::vector<std::vector<double>> layer;
        for (int j = 0; j < numHiddenNodes; ++j){
            std::vector<double> node;
            for(int k = 0; k <= numInputs; ++k){ //FIXME if numInputs =/= numHiddenNodes
                node.push_back(distribution(generator));
            }
            layer.push_back(node);
        }
        weights.push_back(layer);
    }
    
    for (int i = 0; i <= numHiddenNodes; ++i) {
        wHiddenOutput.push_back(distribution(generator));
    }
    
    //////////////////////////////////////////trainer
    
    //initialize deltas
    for (int i = 0; i < numHiddenLayers; ++i) {
        std::vector<std::vector<double>> layer;
        for (int j = 0; j < numHiddenNodes; ++j) {
            std::vector<double> node;
            for (int k = 0; k <= numInputs; ++k) { //FIXME if numInputs =/=numHiddenNodes
                node.push_back(0);
            }
            layer.push_back(node);
        }
        deltaWeights.push_back(layer);
    }
    
    for (int i = 0; i <= numHiddenNodes; ++i) {
        deltaHiddenOutput.push_back(0);
    }
    
    //initialize gradients
    for (int i = 0; i <= numHiddenNodes; ++i) {
        hiddenErrorGradients.push_back(0);
    }
}

neuralNetwork::~neuralNetwork() {
}

inline double neuralNetwork::getOutputErrorGradient(double desiredValue, double outputValue) {
    return (desiredValue - outputValue) / outRange;
}

inline double neuralNetwork::getHiddenErrorGradient(int layer, int neuron) {
    double wGradient = wHiddenOutput[neuron] * outputErrorGradient;
    return hiddenNeurons[layer][neuron] * (1 - hiddenNeurons[layer][neuron]) * wGradient;
}

inline double neuralNetwork::activationFunction(double x) {
    //sigmoid
    if (x < -45) { //from weka, to combat overflow
        x = 0;
    } else if (x > 45) {
        x = 1;
    } else {
        x = 1/(1 + exp(-x));
    }
    return x;
}

double neuralNetwork::process(std::vector<double> inputVector) {
    std::vector<double> pattern;
    for (int h = 0; h < numInputs; h++) {
        pattern.push_back(inputVector[whichInputs[h]]);
    }
    
    //set input layer
    inputNeurons.clear();
    for (int i = 0; i < numInputs; ++i) {
        inputNeurons.push_back((pattern[i] - (inBases[i]) / inRanges[i]));
    }
    inputNeurons.push_back(1);
    
    //calculate hidden layers
    hiddenNeurons.clear();
    for (int i = 0; i < numHiddenLayers; ++i) {
        std::vector<double> layer;
        for (int j=0; j < numHiddenNodes; ++j) {
            layer.push_back(0);
            if (i == 0) {
                for (int k = 0; k <= numInputs; ++k) {
                    layer[j] += inputNeurons[k] * weights[0][j][k];
                }
            } else {
                for (int k = 0; k <= numHiddenNodes; ++k) {
                    layer[j] = hiddenNeurons[i - 1][k] * weights [i][j][k];
                }
            }
            layer[j] = activationFunction(layer[j]);
        }
        layer.push_back(1); //for bias weight
        hiddenNeurons.push_back(layer);
    }
    
    //calculate output
    outputNeuron = 0;
    for (int k=0; k <= numHiddenNodes; ++k){
        outputNeuron += hiddenNeurons[numHiddenLayers - 1][k] * wHiddenOutput[k];
    }
    outputNeuron = (outputNeuron * outRange) + outBase;
    return outputNeuron;
}

void neuralNetwork::train(std::vector<trainingExample> trainingSet) {
    //setup maxes and mins
    std::vector<double> inMax = trainingSet[0].input;
    std::vector<double> inMin = trainingSet[0].input;
    double outMin = trainingSet[0].output;
    double outMax = trainingSet[0].output;
    for (int ti = 1; ti < (int) trainingSet.size(); ++ti) {
        for (int i = 0; i < numInputs; ++i) {
            if (trainingSet[ti].input[i] > inMax[i]) {
                inMax[i] = trainingSet[ti].input[i];
            }
            if (trainingSet[ti].input[i] < inMin[i]) {
                inMin[i] = trainingSet[ti].input[i];
            }
            if (trainingSet[ti].output > outMax) {
                outMax = trainingSet[ti].output;
            }
            if (trainingSet[ti].output < outMin) {
                outMin = trainingSet[ti].output;
            }
        }
    }
    inRanges.clear();
    inBases.clear();
    for (int i = 0; i < numInputs; ++i) {
        inRanges.push_back((inMax[i] - inMin[i])/ 2);
        inBases.push_back((inMax[i] + inMin[i])/ 2);
    }
    outRange = (outMax - outMin)/ 2;
    outBase = (outMax + outMin)/ 2;
    //train
    epoch = 0;
    while (epoch < numEpochs) {
        double incorrectPatterns = 0;
        double mse = 0;
        //run through every training instance
        for (int ti = 0; ti < (int) trainingSet.size(); ++ti) {
            process(trainingSet[ti].input);
            backpropagate(trainingSet[ti].output);
        }
        epoch++;
    }
}

void neuralNetwork::backpropagate(double desiredOutput) {
    
    //deltas between output and hidden
    outputErrorGradient = getOutputErrorGradient(desiredOutput, outputNeuron);
    for (int i = 0; i <= numHiddenNodes; ++i) {
        deltaHiddenOutput[i] = (learningRate * hiddenNeurons[numHiddenLayers - 1][i] * outputErrorGradient) + (momentum * deltaHiddenOutput[i]);
    }
    
    //deltas between hidden
    
    //TODO multiple layers
    
    //deltas input and hidden
    for (int i = 0; i < numHiddenNodes; ++i) {
        hiddenErrorGradients[i] = getHiddenErrorGradient(0, i);
        for (int j = 0; j <= numInputs; ++j) {
            deltaWeights[0][i][j] = (learningRate * inputNeurons[j] * hiddenErrorGradients[i]) + momentum * deltaWeights[0][i][j];
        }
    }
    updateWeights();
}

void neuralNetwork::updateWeights() {
    //input to hidden weights
    for (int i = 0; i < numHiddenNodes; ++i) {
        for (int j = 0; j <= numInputs; ++j) {
            weights[0][i][j] += deltaWeights[0][i][j];
        }
    }
    
    //hidden to hidden weights
    //TODO multiple layers
    
    //hidden to output weights
    for (int i = 0; i <= numHiddenNodes; ++i) {
        wHiddenOutput[i] += deltaHiddenOutput[i];
    }
}
