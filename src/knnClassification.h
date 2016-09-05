#ifndef knnClassification_h
#define knnClassification_h

#include <vector>
#include "baseModel.h"

struct neighbour {
	int classNum;
	std::vector<double> features;
};

class knnClassification : public baseModel {

public:
	knnClassification(int, std::vector<int>, std::vector<neighbour>, int, int, int);
	~knnClassification();
	void addNeighbour(int, std::vector<double>);
	double processInput(std::vector<double>);

private:
	int numInputs;
	std::vector<int> whichInputs;
	std::vector<neighbour> neighbours;
	int numExamples;
	int numNeighbours; //aka "k"
	int numClasses;
	std::pair<int, double>* nearestNeighbours;
};

#endif

