#include <math.h>
#include <algorithm>
#include <vector>
#include "neuralNetwork.h"
#include "nnEmbindings.h"

//#define DEBUG
/*
neuralNetwork::neuralNetwork(int num_inputs, 
                             std::vector<int> which_inputs,
                             int num_hidden_nodes,
                             double*** _weights,
                             std::vector<double> w_hidden_output,
                             std::vector<double> in_max,
                             std::vector<double> in_min,
                             double out_max,
                             double out_min) {
	numInputs = num_inputs;
	whichInputs = which_inputs;
	numHiddenNodes = num_hidden_nodes;
	//input neurons, including bias
	inputNeurons = new double[numInputs + 1];
	for (int i=0; i < numInputs; ++i){
		inputNeurons[i] = 0;
	}
	inputNeurons[numInputs] = 1;

	//hidden neurons, including bias
	hiddenNeurons = new double[numHiddenNodes + 1];
	for (int i=0; i < numHiddenNodes; ++i){
		hiddenNeurons[i] = 0;
	}
	hiddenNeurons[numHiddenNodes] = 1;

	weights = _weights;
	wHiddenOutput = w_hidden_output;

	inRanges = new double[numInputs];
	inBases = new double[numInputs];

	for (int i = 0; i < numInputs; ++i) {
           inRanges[i] = (in_max[i] - in_min[i])/ 2;
           inBases[i] = (in_max[i] + in_min[i])/ 2;
	}

       outRange = (out_max - out_min)/ 2;
       outBase = (out_max + out_min)/ 2;
}
*/
neuralNetwork::neuralNetwork(int num_inputs,
			     std::vector<int> which_inputs,
			     int num_hidden_nodes,
			     std::vector<double> _weights,
			     std::vector<double> w_hidden_output,
			     std::vector<double> in_max,
			     std::vector<double> in_min,
			     double out_max,
			     double out_min
			     ) {
  numInputs = num_inputs;
  
  whichInputs = which_inputs;
  numHiddenNodes = num_hidden_nodes;
  
  //input neurons, including bias
  //inputNeurons = new double[numInputs + 1];
  for (int i=0; i < numInputs; ++i){
    inputNeurons[i] = 0;
  }
  inputNeurons[numInputs] = 1;
  
  //hidden neurons, including bias
  //hiddenNeurons = new double[numHiddenNodes + 1];
  for (int i=0; i < numHiddenNodes; ++i){
    hiddenNeurons[i] = 0;
  }
  hiddenNeurons[numHiddenNodes] = 1;
  
  //winding up a long vector from javascript
  int numLayers = 1;
  int count = 0;
  for (int i = 0; i < numLayers; ++i) {
    std::vector<std::vector<double>> layer;
    for (int j = 0; j < numHiddenNodes; ++j){
      std::vector<double> node;
      for(int k = 0; k <= numInputs; ++k){
	node.push_back( _weights[count]);
#ifdef DEBUG
	printf("weights i %d j %d k %d, count %d, weight %f\n", i, j, k, count, _weights[count]);
#endif
	count++;
      }
      layer.push_back(node);
    }
    weights.push_back(layer);
  }
  
  wHiddenOutput = w_hidden_output;
  
  //inRanges = new double[numInputs];
  //inBases = new double[numInputs];
  
  for (int i = 0; i < numInputs; ++i) {
    inRanges.push_back((in_max[i] - in_min[i])/ 2);
#ifdef DEBUG
    printf("i %d inRanges[i] %f in_max[i] %f in_min[i] %f\n", i, inRanges[i], in_max[i], in_min[i]);
#endif
    inBases.push_back((in_max[i] + in_min[i])/ 2);
  }
  
  outRange = (out_max - out_min)/ 2;
  outBase = (out_max + out_min)/ 2;
  
}

neuralNetwork::~neuralNetwork() {
  //delete[] inputNeurons;
  //delete[] hiddenNeurons;

	//	int maxNodes = std::max(numInputs, numHiddenNodes);
	//	for (int i=0; i <= numInputs; ++i) {
	//		for (int j=0; j <=maxNodes; ++j) {
	//                   delete[] weights[i][j];
	//		}
	//		delete[] weights[i];
	//	}
	//	delete[] weights;

	//delete[] inRanges;
	//delete[] inBases;
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
    pattern[h] = (inputVector[whichInputs[h]]);
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

  //calculate hidden layer
  hiddenNeurons.clear();
  for (int j=0; j < numHiddenNodes; ++j) {
    hiddenNeurons.push_back(0);
    for (int i = 0; i <= numInputs; ++i) {
      hiddenNeurons[j] += inputNeurons[i] * weights[0][j][i];//FIXME: the order here is confusing
#ifdef DEBUG
      printf("inputNeuron %d = %f weight %f j %d\n", i, inputNeurons[i], weights[0][i][j], j);
#endif
    }
#ifdef DEBUG
    printf("pre-hiddenNeuron %d = %f\n", j, hiddenNeurons[j]);
#endif
    hiddenNeurons[j] = activationFunction(hiddenNeurons[j]);
#ifdef DEBUG
    printf("hiddenNeuron %d = %f\n", j, hiddenNeurons[j]);
#endif
  }
  hiddenNeurons.push_back(1);
  //calculate output
  double output = 0;
  for (int k=0; k <= numHiddenNodes; ++k){
    output += hiddenNeurons[k] * wHiddenOutput[k];
  }
  output = (output * outRange) + outBase;
#ifdef DEBUG
  printf("cpp output: %f\n", output);
#endif
  return output;
}

