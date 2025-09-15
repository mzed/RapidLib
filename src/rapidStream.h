/**
 * @file    rapidStream.h
 * @author  Michael Zbyszynski
 * @date    6 Feb 2017
 * @copyright Copyright © 2017 Goldsmiths. All rights reserved.
 */

#ifndef rapidStream_h
#define rapidStream_h

#include <stdint.h>
#include <atomic>
#include "../dependencies/bayesfilter/src/BayesianFilter.h"

#include "rapidlib_export.h"

namespace rapidLib
{
template<typename T>
class rapidStream
{
  public:

  /**
   * Create a circular buffer with 3 elements.
   */
  RAPIDLIB_EXPORT rapidStream();
  /**
   * Create a circular buffer with an arbitrary number of elements.
   * @param int: number of elements to hold in the buffer
   */
  RAPIDLIB_EXPORT rapidStream(std::size_t windowSize);

  RAPIDLIB_EXPORT ~rapidStream();

  /**
   * Resets all the values in the buffer to zero.
   */
  RAPIDLIB_EXPORT void clear();

  /** Add a value to a circular buffer whose size is defined at creation.
   * @param double: value to be pushed into circular buffer.
   */
  RAPIDLIB_EXPORT void pushToWindow(T input);

  /** Calculate the first-order difference (aka velocity) between the last two inputs.
   * @return double: difference between last two inputs.
   */
  RAPIDLIB_EXPORT T velocity() const;

  /** Calculate the second-order difference (aka acceleration) over the last three inputs.
   * @return double: acceleration over the last three inputs.
   */
  RAPIDLIB_EXPORT T acceleration() const;

  /** Find the minimum value in the buffer.
   * @return double: minimum.
   */
  RAPIDLIB_EXPORT T minimum() const;

  /** Find the maximum value in the buffer.
   * @return double: maximum.
   */
  RAPIDLIB_EXPORT T maximum() const;

  /** Count the number of zero crossings in the buffer.
   * @return int: number of zero crossings.
   */
  RAPIDLIB_EXPORT uint32_t numZeroCrossings() const;

  /** Calculate the sum of all values in the buffer.
   * @return T: sum.
   */
  RAPIDLIB_EXPORT T sum() const;

  /** Calculate the mean of all values in the buffer.
   * @return double: mean.
   */
  RAPIDLIB_EXPORT T mean() const;

  /** Calculate the standard deviation of all values in the buffer.
   * @return double: standard deviation.
   */
  RAPIDLIB_EXPORT T standardDeviation() const;

  /** Calculate the root mean square of the values in the buffer
   * @return double: rms
   */
  RAPIDLIB_EXPORT T rms() const;

  /** Non-linear Baysian filtering for EMG envelope extraction.
   * @return current envelope value
   */
  RAPIDLIB_EXPORT T bayesFilter(T inputValue);
  RAPIDLIB_EXPORT void bayesSetDiffusion(float logDiffusion);
  RAPIDLIB_EXPORT void bayesSetJumpRate(float jump_rate);
  RAPIDLIB_EXPORT void bayesSetMVC(float mvc);

  /** Calculate the minimum first-order difference over consecutive inputs in the buffer.
   * @return double: minimum velocity.
   */
  RAPIDLIB_EXPORT T minVelocity() const;

  /** Calculate the maximum first-order difference over consecutive inputs in the buffer.
   * @return double: maximum velocity.
   */
  RAPIDLIB_EXPORT T maxVelocity() const;

  /** Calculate the minimum second-order difference over consecutive inputs in the buffer.
   * @return double: minimum acceleration.
   */
  RAPIDLIB_EXPORT T minAcceleration() const;

  /** Calculate the maximum second-order difference over consecutive inputs in the buffer.
   * @return double: maximum acceleration.
   */
  RAPIDLIB_EXPORT T maxAcceleration() const;

private:
  std::size_t windowSize {};
  std::atomic<uint32_t> windowIndex {};
  std::vector<T> circularWindow {};

  inline T calcCurrentVel(std::size_t i) const;

  BayesianFilter bayesFilt;
};

}; // namespace rapidLib

#endif
