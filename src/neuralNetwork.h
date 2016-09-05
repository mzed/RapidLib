#ifndef neuralNetwork_h
#define neuralNetwork_h

#include <vector>
#include "baseModel.h"

class neuralNetwork : public baseModel {

public:

  //neuralNetwork(int, std::vector<int>, int, double***, std::vector<double>, std::vector<double>, std::vector<double>, double, double);
  neuralNetwork(int, std::vector<int>, int, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double, double);
  //neuralNetwork(int, std::vector<int>, int, std::vector<double>, std::vector<double>);
  ~neuralNetwork();

  double processInput(std::vector<double>);

private:

  int numInputs;
  std::vector<int> whichInputs;
  
  int numHiddenNodes;
  
  std::vector<double> inputNeurons;
  std::vector<double> hiddenNeurons;
  std::vector<std::vector<std::vector<double>>> weights;
  std::vector<double> wHiddenOutput;
  
  std::vector<double> inRanges;
  std::vector<double> inBases;
  double outRange;
  double outBase;

  inline double activationFunction(double);
};

#endif

