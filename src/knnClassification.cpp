#include <math.h>
#include <utility>
#include "knnClassification.h"
#include "knnEmbindings.h"

knnClassification::knnClassification(int num_inputs, std::vector<int> which_inputs, std::vector<neighbour> _neighbours, int num_examples, int k, int num_classes) {
	numInputs = num_inputs;
	whichInputs = which_inputs;
	numExamples = num_examples;
	neighbours = _neighbours;
	numNeighbours = k;
	nearestNeighbours = new std::pair<int, double>[numNeighbours];
	numClasses = num_classes;
}

knnClassification::~knnClassification() {
	delete[] nearestNeighbours;
}

void knnClassification::addNeighbour(int classNum, std::vector<double> features) {
  neighbour newNeighbour = {classNum, features};
  neighbours.push_back(newNeighbour);
};

double knnClassification::processInput(std::vector<double> inputVector) {
   for (int i = 0; i < numNeighbours; ++i) {
     nearestNeighbours[i] = {0, 0.};
   };
   std::pair<int, double> farthestNN = {0, 0.};

   double pattern[numInputs];
   
   for (int h = 0; h < numInputs; h++) {
     pattern[h] = inputVector[whichInputs[h]];
   }

   //Find k nearest neighbours
   for (int i = 0; i < numExamples; ++i) {
     //find Euclidian distance for this neighbor
     double euclidianDistance = 0;
     for(int j = 0; j < numInputs ; ++j){
       euclidianDistance = euclidianDistance + pow((pattern[j] - neighbours[i].features[j]),2);
     }
     euclidianDistance = sqrt(euclidianDistance);
     if (i < numNeighbours) {
       //save the first k neighbours
       nearestNeighbours[i] = {i, euclidianDistance};
       if (euclidianDistance > farthestNN.second) {
	 farthestNN = {i, euclidianDistance};
       }
     } else if (euclidianDistance < farthestNN.second) {
       //replace farthest, if new neighbour is closer
       nearestNeighbours[farthestNN.first] = {i, euclidianDistance};
       int currentFarthest = 0;
       double currentFarthestDistance = 0.;
       for (int n = 0; n < numNeighbours; n++) {
	 if (nearestNeighbours[n].second > currentFarthestDistance) {
	   currentFarthest = n;
	   currentFarthestDistance = nearestNeighbours[n].second;
	 }
       }
       farthestNN = {currentFarthest, currentFarthestDistance};
     }
    }

   //majority vote on nearest neighbours
   std::vector<int> numVotesPerClass;
   for (int i=0; i < numClasses; ++i) {
     numVotesPerClass.push_back(0);
   }
   for (int i = 0; i < numNeighbours; ++i){
     numVotesPerClass[neighbours[nearestNeighbours[i].first].classNum - 1]++;
   }
   double foundClass = 0;
   int mostVotes = 0;
   for (int i = 0; i < numClasses; ++i) {
     if (numVotesPerClass[i] > mostVotes) { //TODO: Handle ties the same way Wekinator does
       mostVotes = numVotesPerClass[i];
       foundClass = i + 1;
     }
   }
   return foundClass;
}
