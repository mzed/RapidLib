#include <cmath>
#include <utility>
#include <map>
#include <vector>
#include "knnClassification.h"
//#include "knnEmbindings.h"

knnClassification::knnClassification(int num_inputs, std::vector<int> which_inputs, std::vector<trainingExample> _neighbours, int k)
  : numInputs(num_inputs),
    whichInputs(which_inputs),
    neighbours(_neighbours),
    numNeighbours(k)
{
  nearestNeighbours = new std::pair<int, double>[numNeighbours];
}

knnClassification::~knnClassification() {
	delete[] nearestNeighbours;
}

void knnClassification::addNeighbour(int classNum, std::vector<double> features) {
    trainingExample newNeighbour = {features, std::vector<double>(classNum)};
  neighbours.push_back(newNeighbour);
};

void knnClassification::train(std::vector<trainingExample> trainingSet) {
  neighbours.clear();
  neighbours = trainingSet;
};

double knnClassification::process(std::vector<double> inputVector) {
   for (int i = 0; i < numNeighbours; ++i) {
     nearestNeighbours[i] = {0, 0.};
   };
   std::pair<int, double> farthestNN = {0, 0.};

   double pattern[numInputs];
   
   for (int h = 0; h < numInputs; h++) {
     pattern[h] = inputVector[whichInputs[h]];
   }

   //Find k nearest neighbours
   int index = 0;
   for (std::vector<trainingExample>::iterator it = neighbours.begin(); it != neighbours.end(); ++it) {
     //find Euclidian distance for this neighbor
     double euclidianDistance = 0;
     for(int j = 0; j < numInputs ; ++j){
         euclidianDistance = euclidianDistance + pow((pattern[j] - it->input[j]), 2);
     }
     euclidianDistance = sqrt(euclidianDistance);
     if (index < numNeighbours) {
       //save the first k neighbours
       nearestNeighbours[index] = {index, euclidianDistance};
       if (euclidianDistance > farthestNN.second) {
	 farthestNN = {index, euclidianDistance};
       }
     } else if (euclidianDistance < farthestNN.second) {
       //replace farthest, if new neighbour is closer
       nearestNeighbours[farthestNN.first] = {index, euclidianDistance};
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
     ++index;
   }

   //majority vote on nearest neighbours
   std::map<int, int> classVoteMap;
   typedef std::pair<int, int> classVotePair;
   for (int i = 0; i < numNeighbours; ++i){
       int classNum = std::round(neighbours[nearestNeighbours[i].first].output[0]);
     if ( classVoteMap.find(classNum) == classVoteMap.end() ) {
       classVoteMap.insert(classVotePair(classNum, 1));
     } else {
       classVoteMap[classNum]++;
     }
   }
   double foundClass = 0;
   int mostVotes = 0;
   std::map<int, int>::iterator p;
   for(p = classVoteMap.begin(); p != classVoteMap.end(); ++p)
     {
       if (p->second > mostVotes) {
	 mostVotes = p->second;
	 foundClass = p->first;
       }
     }
   return foundClass;
}
