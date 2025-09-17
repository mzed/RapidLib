/**
 * @file seriesClassification.h
 * RapidLib
 *
 * @author Michael Zbyszynski
 * @date 08 Jun 2017
 * @copyright Copyright © 2017 Goldsmiths. All rights reserved.
 */

#pragma once

#include "rapidLib/rapidLib_export.h"

#include "../../src/fastDTW.h"
#include "trainingExample.h"

#include <vector>
#include <string>
#include <map>

/** Class for containing time series classifiers.
 *
 * Currently only (fast)DTW.
 */

template<typename T>
class seriesClassificationTemplate final
{
public:
  
  /** Constructor, no params */
  RAPIDLIB_EXPORT seriesClassificationTemplate();
  RAPIDLIB_EXPORT ~seriesClassificationTemplate();
  
  /**  Train on a specified set of trainingSeries
   * @param std::vector<trainingSeries> A vector of training series
   */
  RAPIDLIB_EXPORT bool train(const std::vector<trainingSeriesTemplate<T> > &seriesSet);

  /** Reset model to its initial state, forget all costs and training data*/
  RAPIDLIB_EXPORT void reset();
  
  /** Compare an input series to the stored training series
   * @param std::vector<std::vector> vector of vectors, either float or double input data
   * @return The label of the closest training series.
   */
   RAPIDLIB_EXPORT std::string run(const std::vector<std::vector<T> > &inputSeries);

  /** Compare an input series to all of the stored series with a specified label
   * @param std::vector<std::vector> either float or double input data
   * @param String label to compare with
   * @return The lowest cost match, float or double
   */
  RAPIDLIB_EXPORT T run(const std::vector<std::vector<T> >& inputSeries, std::string label);

  /** Compare an input series to the stored training series. Parallel processing
   * @param std::vector<std::vector> vector of vectors, either float or double input data
   * @return The label of the closest training series.
   */
  RAPIDLIB_EXPORT std::string runParallel(const std::vector<std::vector<T> >& inputSeries);
  
  /** Compare an input series to all of the stored series with a specified label. Parallel processing
   * @param std::vector<std::vector> either float or double input data
   * @param String label to compare with
   * @return The lowest cost match, float or double
   */
  RAPIDLIB_EXPORT T runParallel(const std::vector<std::vector<T> > &inputSeries, std::string label);

  /** Compare an input series to all of the stored series with a specified label
   * @param std::vector<T> one frame either float or double input data
   * @return The lowest cost match, float or double
   */
  RAPIDLIB_EXPORT std::string runContinuous(const std::vector<T> &inputVector);

  /** Get the costs that were calculated by the run method
   * @return A vector of floats or doubles, the cost of matching to each training series
   */
  RAPIDLIB_EXPORT std::vector<T> getCosts() const;

  /** Get minimum training series length
   * @return The minimum length training series
   */
  RAPIDLIB_EXPORT std::size_t getMinLength() const;

  /** Get minimum training series length from a specified label
   * @param string The label to check
   * @return The minimum length training series of that label
   */
  RAPIDLIB_EXPORT std::size_t getMinLength(std::string label) const;

  /** Get maximum training series length
   * @return The maximum length training series
   */
  RAPIDLIB_EXPORT std::size_t getMaxLength() const;

  /** Get maximum training series length from a specified label
   * @param string The label to check
   * @return The maximum length training series of that label
   */
  RAPIDLIB_EXPORT std::size_t getMaxLength(std::string label) const;

  /** Return struct for calculate costs */
  template<typename TT>
  struct minMax {
    TT min;
    TT max;
  };
  
  /** Calculate minimum and maximum cost between examples in a label.
   * @param string Label to calculate
   * @return minMax struct containing min and max
   */
  RAPIDLIB_EXPORT minMax<T> calculateCosts(std::string label) const;

  /** Calculate minimum and maximum cost between examples in one label and examples in a second.
   * @param string first label to compare
   * @param string second label to compare
   * @return minMax struct containing min and max
   */
  RAPIDLIB_EXPORT minMax<T> calculateCosts(std::string label1, std::string label2) const;

private:
  std::vector<trainingSeriesTemplate<T> > allTrainingSeries;
  size_t vectorLength;
  std::vector<T> allCosts;
  size_t maxLength;
  size_t minLength;
  std::map<std::string, minMax<size_t> > lengthsPerLabel;
  bool isTraining { false };
  
  std::vector<std::vector<T> > seriesBuffer;
  int hopSize {1};
  int counter {};
  
  size_t findClosestSeries() const;
  void runThread(const std::vector<std::vector<T>> &inputSeries, std::size_t i);
};

namespace rapidLib
{
//This is here to keep the old API working
using seriesClassification = seriesClassificationTemplate<double>;
using seriesClassificationFloat = seriesClassificationTemplate<float>;
}
