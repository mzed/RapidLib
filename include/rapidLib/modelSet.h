/**
 * @file modelSet.h
 * RapidLib
 *
 * @author Michael Zbyszynski
 * @date 26 Sep 2016
 * @copyright Copyright © 2016 Goldsmiths. All rights reserved.
 */

#pragma once

#include "rapidLib/rapidLib_export.h"
#include "trainingExample.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

template<typename T> class baseModel;

#ifndef EMSCRIPTEN
namespace Json { class Value; }
#endif

/** This class holds a set of models with the same or different algorithms. */
template<typename T>
class modelSet
{
public:
  RAPIDLIB_EXPORT modelSet();
  RAPIDLIB_EXPORT virtual ~modelSet();

  // Move-only: models are owned through unique_ptr. Defined in modelSet.cpp, where baseModel is complete.
  modelSet(const modelSet&) = delete;
  modelSet& operator=(const modelSet&) = delete;
  RAPIDLIB_EXPORT modelSet(modelSet&&) noexcept;
  RAPIDLIB_EXPORT modelSet& operator=(modelSet&&) noexcept;

  /** Train on a specified set, causes creation if not created */
  RAPIDLIB_EXPORT virtual bool train(const std::vector<trainingExampleTemplate<T> > &trainingSet);

  /** reset to pre-training state */
  RAPIDLIB_EXPORT bool reset();

  /** Generate an output value from a single input vector.
   *
   * Will return an error if training in progress.
   *
   * @param vector A standard vector of type T that is the input for classification or regression.
   * @return vector A vector type T that are the predictions for each model in the set.
   */
  RAPIDLIB_EXPORT std::vector<T> run(const std::vector<T> &inputVector) const;

protected:
  std::vector<std::unique_ptr<baseModel<T>>> models{};
  std::optional<size_t> numInputs{};
  std::vector<std::string> inputNames{};
  std::optional<size_t> numOutputs{};
  bool isTraining{}; //This is true while the models are training, and will block running
  bool isTrained{};

  void threadTrain(std::size_t i, const std::vector<trainingExampleTemplate<T> >& training_set);

#ifndef EMSCRIPTEN //The javascript code will do its own JSON parsing
public:

  /** Get a JSON representation of the model
   *
   * @return Styled string JSON representation
   */
  RAPIDLIB_EXPORT std::string getJSON() const;

  /** Write a JSON model description to specified file path
   *
   * @param file path
   *
   */
  RAPIDLIB_EXPORT void writeJSON(const std::string &filepath) const;

  /** configure empty model with string. See getJSON() */
  RAPIDLIB_EXPORT bool putJSON(const std::string &jsonMessage);

  /** read a JSON file at file path and build a modelSet from it */
  RAPIDLIB_EXPORT bool readJSON(const std::string &filepath);

private:
  Json::Value parse2json() const;
  void json2modelSet(const Json::Value &root);
#endif
};
