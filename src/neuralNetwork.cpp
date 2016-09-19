#include <math.h>
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include "neuralNetwork.h"

//#define DEBUG

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
    //winding up a long vector from javascript
    int count = 0;
    for (int i = 0; i < numHiddenLayers; ++i) {
        std::vector<std::vector<double>> layer;
        for (int j = 0; j < numHiddenNodes; ++j){
            std::vector<double> node;
            for(int k = 0; k <= numInputs; ++k){ //FIXME if numInputs =/=numHiddenNodes
                if (randomize) {
                    node.push_back( ((double)rand()/RAND_MAX) - 0.5);
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
            wHiddenOutput.push_back(((double)rand()/RAND_MAX) - 0.5);
        }
    }
    
    for (int i = 0; i < numInputs; ++i) {
        inRanges.push_back((in_max[i] - in_min[i])/ 2);
#ifdef DEBUG
        printf("i %d inRanges[i] %f in_max[i] %f in_min[i] %f\n", i, inRanges[i], in_max[i], in_min[i]);
#endif
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

neuralNetwork::~neuralNetwork() {
}

inline double neuralNetwork::getOutputErrorGradient(double desiredValue, double outputValue) {
    return outputValue * (1 - outputValue) * (desiredValue - outputValue);
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

double neuralNetwork::processInput(std::vector<double> inputVector) {
    double pattern[numInputs];
    for (int h = 0; h < numInputs; h++) {
        pattern[h] = inputVector[h]; //(inputVector[whichInputs[h]]);
#ifdef DEBUG
        printf("pattern %d = %f\n", h, inputVector[whichInputs[h]]);
#endif
    }
    
    //set input layer
    inputNeurons.clear();
    for (int i = 0; i < numInputs; ++i) {
        inputNeurons.push_back((pattern[i] - inBases[i]) / inRanges[i]);
#ifdef DEBUG
        printf("pattern %f, base %f, range %f\n", pattern[i], inBases[i], inRanges[i]);
        printf("inputNeuron %d = %f\n", i, inputNeurons[i]);
#endif
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
#ifdef DEBUG
        printf("hidden Neuron %f - weight %f\n", hiddenNeurons[k], wHiddenOutput[k]);
        printf("summing output: %f\n", output);
#endif
    }
    outputNeuron = (outputNeuron * outRange) + outBase;
#ifdef DEBUG
    printf("cpp output: %f\n", outputNeuron);
#endif
    return outputNeuron;
}

void neuralNetwork::train(std::vector<trainingExample> trainingSet) {
    epoch = 0;
    while (epoch < numEpochs) {
        double incorrectPatterns = 0;
        double mse = 0;
        //run through every training instance
        for (int ti = 0; ti < (int) trainingSet.size(); ++ti) {
            processInput(trainingSet[ti].input);
            backpropagate(trainingSet[ti].output);
        }
        epoch++;
        std::cout << "epoch: " << epoch << std::endl;
    }
}

void neuralNetwork::backpropagate(double desiredOutput) {
    
    //deltas between output and hidden
    std::cout << "desired: " << desiredOutput << std::endl;
    std::cout << "outputNeuron: " << outputNeuron << std::endl;
    outputErrorGradient = getOutputErrorGradient(desiredOutput, outputNeuron);
    std::cout << "oEG " << outputErrorGradient << std::endl;
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
