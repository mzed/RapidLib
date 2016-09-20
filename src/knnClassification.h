#ifndef knnClassification_h
#define knnClassification_h

#include <vector>
#include "baseModel.h"


class knnClassification : public baseModel {

 public:
  knnClassification(int, std::vector<int>, std::vector<trainingExample>, int);
  ~knnClassification();
  void addNeighbour(int, std::vector<double>);
  double process(std::vector<double>);
  void train(std::vector<trainingExample>);
  
 private:
  int numInputs;
  std::vector<int> whichInputs;
  std::vector<trainingExample> neighbours;
  int numNeighbours; //aka "k"
  std::pair<int, double>* nearestNeighbours;

};

#endif

