//
//  knnClassification.cpp
//  RapidLib
//
//  Created by mzed on 05/09/2016.
//  Copyright © 2016 Goldsmiths. All rights reserved.
//

#include <math.h>
#include <utility>
#include <map>
#include <vector>
#include <algorithm>
#include "knnClassification.h"
#ifdef EMSCRIPTEN
#include "emscripten/knnEmbindings.h"
#endif

template<typename T>
knnClassification<T>::knnClassification(const int &num_inputs,
                                        const std::vector<size_t> &which_inputs,
                                        const std::vector<trainingExampleTemplate<T> > &_neighbours,
                                        const int k) :
numInputs(num_inputs),
whichInputs(which_inputs),
whichOutput(0),
neighbours(_neighbours),
desiredK(k),
currentK(k)
{
}

template<typename T>
knnClassification<T>::~knnClassification() {}

template<typename T>
void knnClassification<T>::reset()
{
  //TODO: implement this
}

template<typename T>
size_t knnClassification<T>::getNumInputs() const
{
  return numInputs;
}

template<typename T>
std::vector<size_t> knnClassification<T>::getWhichInputs() const
{
  return whichInputs;
}

template<typename T>
int knnClassification<T>::getK() const
{
  return currentK;
}

template<typename T>
inline void knnClassification<T>::updateK()
{
  if (currentK != desiredK) currentK = std::min(desiredK, (int) neighbours.size());
}

template<typename T>
void knnClassification<T>::setK(int newK)
{
  desiredK = newK;
  updateK();
}

template<typename T>
void knnClassification<T>::addNeighbour(const int &classNum, const std::vector<T> &features)
{
  std::vector<T> classVec {};
  classVec.push_back(T(classNum));
  trainingExampleTemplate<T>  newNeighbour = {features, classVec};
  neighbours.push_back(newNeighbour);
  updateK();
};

template<typename T>
void knnClassification<T>::train(const std::vector<trainingExampleTemplate<T> >& trainingSet)
{
  train(trainingSet, 0);
}

// FIXME: Not paying attention to whichOutput.
template<typename T>
void knnClassification<T>::train(const std::vector<trainingExampleTemplate<T> > &trainingSet, const std::size_t which_output) //FIXME: Does numInputs need to be reset here? -MZ
{
  neighbours.clear();
  neighbours = trainingSet;
  updateK();
  whichOutput = which_output;
};

template<typename T>
T knnClassification<T>::run(const std::vector<T> &inputVector)
{
  std::vector<std::pair<int, T>> nearestNeighbours {}; //These are our k nearest neighbours
  
  for (size_t i {}; i < currentK; ++i)
  {
    nearestNeighbours.push_back( std::make_pair(0, 0.) );
  };
  
  std::pair<int, T> farthestNN {0, 0.}; //This one will be replaced if there's a closer one
  std::vector<T> pattern; //This is what we're trying to match
  
  for (size_t h {}; h < numInputs; ++h)
  {
    pattern.push_back(inputVector[whichInputs[h]]);
  }
  
  //Find k nearest neighbours
  size_t index {};
  
  for (auto it = neighbours.cbegin(); it != neighbours.cend(); ++it)
  {
    //find Euclidian distance for this neighbor
    T euclidianDistance {};
    
    for(size_t j {}; j < numInputs ; ++j)
    {
      euclidianDistance += (T)pow((pattern[j] - it->input[j]), 2);
    }
    
    euclidianDistance = sqrt(euclidianDistance);
    
    if (index < currentK)
    {
      //save the first k neighbours
      nearestNeighbours[index] = {index, euclidianDistance};
      if (euclidianDistance > farthestNN.second) farthestNN = {index, euclidianDistance};
    }
    else if (euclidianDistance < farthestNN.second)
    {
      //replace farthest, if new neighbour is closer
      nearestNeighbours[farthestNN.first] = {index, euclidianDistance};
      size_t currentFarthest {};
      T currentFarthestDistance = 0.;
      
      for (size_t n {}; n < currentK; ++n)
      {
        if (nearestNeighbours[n].second > currentFarthestDistance)
        {
          currentFarthest = n;
          currentFarthestDistance = nearestNeighbours[n].second;
        }
      }
      
      farthestNN = { currentFarthest, currentFarthestDistance} ;
    }
    ++index;
  }
  
  //majority vote on nearest neighbours
  std::map<T, int> classVoteMap;
  using classVotePair = std::pair<int, int>;
  
  for (size_t i {}; i < currentK; ++i)
  {
    T classNum = (T)round(neighbours[nearestNeighbours[i].first].output[whichOutput]);
    if ( classVoteMap.find(classNum) == classVoteMap.end() )
    {
      classVoteMap.insert(classVotePair(classNum, 1));
    }
    else
    {
      ++classVoteMap[classNum];
    }
  }
  
  T foundClass {};
  int mostVotes {};
  
  for (auto const& [whichClass, votes] : classVoteMap)
  {
    if (votes > mostVotes)
    {
      mostVotes = votes;
      foundClass = whichClass;
    }
  }
  
  return foundClass;
}

#ifndef EMSCRIPTEN
template<typename T>
void knnClassification<T>::getJSONDescription(Json::Value &jsonModelDescription)
{
  jsonModelDescription["modelType"] = "kNN Classificiation";
  jsonModelDescription["numInputs"] = numInputs;
  jsonModelDescription["whichInputs"] = this->vector2json(whichInputs);
  jsonModelDescription["k"] = desiredK;
  Json::Value examples;
  
  for (auto const& neighbour : neighbours)
  //for (auto it = neighbours.cbegin(); it != neighbours.cend(); ++it)
  {
    Json::Value oneExample;
    oneExample["class"] = neighbour.output[whichOutput];
    oneExample["features"] = this->vector2json(neighbour.input);
    examples.append(oneExample);
  }
  
  jsonModelDescription["examples"] = examples;
}
#endif

//explicit instantiation
template class knnClassification<double>;
template class knnClassification<float>;
