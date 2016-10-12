#ifndef neuralNetwork_h
#define neuralNetwork_h
#include <vector>
#include "baseModel.h"

#define LEARNING_RATE 0.3
#define MOMENTUM 0.2
#define NUM_EPOCHS 500

/*! Class for implementing a Neural Network.
 *
 * This class includes both running and training, and constructors for reading trained models from JSON.
 */
class neuralNetwork : public baseModel {
    
public:
    /** This is the constructor for building a trained model from JSON. */
    neuralNetwork(int num_inputs, std::vector<int> which_inputs, int num_hidden_layers, int num_hidden_nodes, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double, double);
    
    /** This constructor creates a neural network that needs to be trained.
     *
     * @param num_inputs is the number of inputs the network will process
     * @param which_inputs is an vector of which values in the input vector are being fed to the network. ex: {0,2,4} 
     * @param num_hidden_layer is the number of hidden layers in the network. Must be at least 1.
     * @param num_hidden_nodes is the number of hidden nodes in each hidden layer. Often, this is the same as num_inputs
     *
     * @return A neuralNetwork instance with randomized weights and no normalization values. These will be set or adjusted during training.
     */
    neuralNetwork(int num_inputs, std::vector<int> which_inputs, int num_hidden_layer, int num_hidden_nodes);
    ~neuralNetwork();
    
    /** Generate an output value from a single input vector. 
     * @param A standard vector of doubles that feed-forward regression will run on.
     * @return A single double, which is the result of the feed-forward operation
     */
    double process(std::vector<double> inputVector);
    
private:
    /** Parameters that describe the topography of the model */
    int numInputs;
    std::vector<int> whichInputs;
    int numHiddenLayers;
    int numHiddenNodes;
    
    /** Neurons: state is updated on each process(). */
    std::vector<double> inputNeurons;
    std::vector<std::vector<double>> hiddenNeurons;
    double outputNeuron;
    
    /** Weights between layers and nodes are kept here. */
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<double> wHiddenOutput;
    
    /** Normalization parameters */
    std::vector<double> inRanges;
    std::vector<double> inBases;
    double outRange;
    double outBase;
    
    /** Sigmoid function for activating hidden nodes. */
    inline double activationFunction(double);
    
    /** These pertain to the training, and aren't need to run a trained model */
public:
    /** Train a model using backpropagation.
     *
     * @param The training set is a vector of training examples that contain both a vector of input values and a double specifying desired output.
     *
     */
    void train(std::vector<trainingExample> trainingSet);
    
private:
    /** Parameters that influence learning */
    double learningRate;
    double momentum;
    int numEpochs;
    
    /** These deltas are applied to the weights in the network */
    std::vector<std::vector< std::vector<double>>> deltaWeights;
    std::vector<double> deltaHiddenOutput;
    
    /** Parameters and functions for calculating amount of change for each weight */
    std::vector<double> hiddenErrorGradients;
    double outputErrorGradient;
    inline double getHiddenErrorGradient(int, int);
    
    /** Propagate output error back through the network. 
     * @param The desired output of the network is fed into the function, and compared with the actual output
     */
    void backpropagate(double);
    
    /** Apply corrections to network weights, based on output error */
    void updateWeights();
};

#endif
