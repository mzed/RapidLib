//
//  classification.cpp
//  RapidLib
//
//  Created by mzed on 26/09/2016.
//  Copyright © 2016 Goldsmiths. All rights reserved.
//

#include "rapidLib/classification.h"
#include "knnClassification.h"
#include "svmClassification.h"
#ifdef EMSCRIPTEN
#include "emscripten/classificationEmbindings.h"
#endif

#include <stdexcept>
#include <vector>

template<typename T>
classificationTemplate<T>::classificationTemplate()
{
  modelSet<T>::numInputs = -1;
  modelSet<T>::numOutputs = -1;
  modelSet<T>::isTraining = false;
  classificationType = knn; //this is the default algorithm
};

template<typename T>
classificationTemplate<T>::classificationTemplate(classificationTypes classification_type) :
classificationType { classification_type }
{
  modelSet<T>::numInputs = -1;
  modelSet<T>::numOutputs = -1;
  modelSet<T>::isTraining = false;
};

template<typename T>
classificationTemplate<T>::classificationTemplate(const int num_inputs, const int num_outputs) //TODO: this feature isn't really useful
{
  modelSet<T>::numInputs = num_inputs;
  modelSet<T>::numOutputs = num_outputs;
  modelSet<T>::isTraining = false;
  std::vector<size_t> whichInputs {};

  for (size_t inputIndex {}; inputIndex < modelSet<T>::numInputs; ++inputIndex)
  {
    whichInputs.push_back(inputIndex);
  }
  std::vector<trainingExampleTemplate<T> > trainingSet;

  modelSet<T>::models.clear();
  for (int i{}; i < modelSet<T>::numOutputs; ++i)
  {
    modelSet<T>::models.push_back(std::make_unique<knnClassification<T>>(modelSet<T>::numInputs, whichInputs, trainingSet, 1));
  }
};

template<typename T>
classificationTemplate<T>::classificationTemplate(const std::vector<trainingExampleTemplate<T> > &trainingSet)
{
  modelSet<T>::numInputs = -1;
  modelSet<T>::numOutputs = -1;
  modelSet<T>::isTraining = false;
  train(trainingSet);
};

template<typename T>
bool classificationTemplate<T>::train(const std::vector<trainingExampleTemplate<T> > &training_set)
{
  //TODO: time this process?
  modelSet<T>::reset();
  
  if (training_set.size() > 0)
  {
    //create model(s) here
    modelSet<T>::numInputs = static_cast<int>(training_set[0].input.size());
    modelSet<T>::numOutputs = static_cast<int>(training_set[0].output.size());

    for (int i {}; i < modelSet<T>::numInputs; ++i)
    {
      modelSet<T>::inputNames.push_back("inputs-" + std::to_string(i + 1));
    }

    modelSet<T>::numOutputs = static_cast<int>(training_set[0].output.size());

    for (const auto& example : training_set)
    {
      if (example.input.size() != modelSet<T>::numInputs)
      {
        throw std::length_error("unequal feature vectors in input.");
        return false;
      }

      if (example.output.size() != modelSet<T>::numOutputs)
      {
        throw std::length_error("unequal output vectors.");
        return false;
      }
    }

    std::vector<size_t> whichInputs {};

    for (int inputNum {}; inputNum < modelSet<T>::numInputs; ++inputNum)
    {
      whichInputs.push_back(inputNum);
    }
    
    for (int outputNum {}; outputNum < modelSet<T>::numOutputs; ++outputNum)
    {
      if (classificationType == svm)
      {
        modelSet<T>::models.push_back(std::make_unique<svmClassification<T>>(modelSet<T>::numInputs));
      }
      else
      {
        modelSet<T>::models.push_back(std::make_unique<knnClassification<T>>(modelSet<T>::numInputs, whichInputs, training_set, 1));
      }
    }
    
    return modelSet<T>::train(training_set);
  }
  return false;
}

template<typename T>
std::vector<int> classificationTemplate<T>::getK() const
{
  std::vector<int> kVector {};

  for (const auto& model : modelSet<T>::models)
  {
    kVector.push_back(dynamic_cast<const knnClassification<T>*>(model.get())->getK()); //FIXME: I really dislike this design
  }

  return kVector;
}

template<typename T>
void classificationTemplate<T>::setK(const int whichModel, const int newK)
{
  if (modelSet<T>::models.size() > whichModel)
  {
    dynamic_cast<knnClassification<T>*>(modelSet<T>::models[whichModel].get())->setK(newK); //FIXME: I really dislike this design
  }
  else
  {
    throw std::length_error("model not in model set.");
  }
}

//explicit instantiation
template class classificationTemplate<double>;
template class classificationTemplate<float>;
