#ifndef neuralNetwork_h
#define neuralNetwork_h

#include <vector>
#include "baseModel.h"

#define LEARNING_RATE 0.3
#define MOMENTUM 0.2
#define NUM_EPOCHS 500

class neuralNetwork : public baseModel {
    
public:
    neuralNetwork(int, std::vector<int>, int, int, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double, double);
    neuralNetwork(int, std::vector<int>, int, int);
    ~neuralNetwork();
    
    double process(std::vector<double>);
    
private:
    int numInputs;
    std::vector<int> whichInputs;
    
    int numHiddenLayers;
    int numHiddenNodes;
    
    //neurons
    std::vector<double> inputNeurons;
    std::vector<std::vector<double>> hiddenNeurons;
    double outputNeuron;
    
    //weights
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<double> wHiddenOutput;
    
    //for normalization
    std::vector<double> inRanges;
    std::vector<double> inBases;
    double outRange;
    double outBase;
    
    inline double activationFunction(double);
    
    //trainer
public:
    void train(std::vector<trainingExample>);
    
private:
    //learning
    double learningRate;
    double momentum;
    int epoch;
    int numEpochs;
    
    //changes
    std::vector<std::vector< std::vector<double>>> deltaWeights;
    std::vector<double> deltaHiddenOutput;
    
    //error gradients
    std::vector<double> hiddenErrorGradients;
    double outputErrorGradient;
    inline double getOutputErrorGradient(double, double);
    inline double getHiddenErrorGradient(int, int);
    
    void backpropagate(double);
    void updateWeights();
};

#endif
