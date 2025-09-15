/**
 * @file regression.h
 * RapidLib
 *
 * @author Michael Zbsyzynski
 * @date 26 Sep 2016
 * @copyright Copyright © 2016 Goldsmiths. All rights reserved.
 */

#ifndef REGRESSION_H
#define REGRESSION_H

#include <vector>
#include "modelSet.h"

/*! Class for implementing a set of regression models.
 *
 * This doesn't do anything modelSet can't do. But, it's simpler and more like wekinator.
 * It has some calls that are specifc to neural networks
 */

template<typename T>
class regressionTemplate final : public modelSet<T>
{
public:
  /** with no arguments, just make an empty vector */
  RAPIDLIB_EXPORT regressionTemplate();
  /** create based on training set inputs and outputs */
  RAPIDLIB_EXPORT regressionTemplate(const std::vector<trainingExampleTemplate<T> > &trainingSet);
  /** create with proper models, but not trained */
  RAPIDLIB_EXPORT regressionTemplate(const int &numInputs, const int &numOutputs);

  /** destructor */
  ~regressionTemplate() {};
  
  /** Train on a specified set, causes creation if not created */
  RAPIDLIB_EXPORT bool train(const std::vector<trainingExampleTemplate<T> > &trainingSet) override;

  /** Check how far the training has gotten. Averages progress over all models in training */
  RAPIDLIB_EXPORT float getTrainingProgress();

  /** Check how many training epochs each model will run. This feature is temporary, and will be replaced by a different design. */
  RAPIDLIB_EXPORT std::vector<size_t> getNumEpochs() const;

  /** Call before train, to set the number of training epochs */
  RAPIDLIB_EXPORT void setNumEpochs(const size_t &epochs);

  /** Check how many hidden layers are in each model. This feature is temporary, and will be replaced by a different design. */
  RAPIDLIB_EXPORT std::vector<size_t> getNumHiddenLayers() const;

  /** Set how many hidden layers are in all models. This feature is temporary, and will be replaced by a different design. */
  RAPIDLIB_EXPORT void setNumHiddenLayers(const int &num_hidden_layers);
  
  /** Check how many hidden nodes are in each model. This feature is temporary, and will be replaced by a different design. */
  RAPIDLIB_EXPORT std::vector<size_t> getNumHiddenNodes() const;

  /** Set how many hidden layers are in all models. This feature is temporary, and will be replaced by a different design. */
  RAPIDLIB_EXPORT void setNumHiddenNodes(const int &num_hidden_nodes);

private:
  size_t numEpochs { 500 }; //Temporary -- also should be part of nn only. -mz
  size_t numHiddenLayers { 1 }; //Temporary -- this should be part of the nn class. -mz
  size_t numHiddenNodes; //Temporary -- also should be part of nn only. -mz
  bool created;
};

namespace rapidLib
{
//This is here so the old API still works
using regression = regressionTemplate<double>;
using regressionFloat = regressionTemplate<float>;
};

#endif
