/**
 * @file fastDTW.cpp
 * RapidLib
 *
 * @author Michael Zbyszynski
 * @date 07 Sep 2017
 * @copyright Copyright © 2017 Goldsmiths. All rights reserved.
 */

#include "fastDTW.h"
#include "dtw.h"

namespace {

template<typename T>
static std::vector<std::vector<T> > downsample(const std::vector<std::vector<T>>& series)
{
  std::vector<std::vector<T> > shrunkenSeries {};

  for (std::size_t i {}; i < series.size(); ++i)
  {
    if (i % 2 == 0)
    {
      shrunkenSeries.push_back(series[i]);
    }
    else
    {
      const int shrunkIndex { static_cast<int>(i * 0.5) };

      for (std::size_t j {}; j < series[i].size(); ++j)
      {
        shrunkenSeries[shrunkIndex][j] = (shrunkenSeries[shrunkIndex][j] + series[i][j]) * (T)0.5;
      }
    }
  }

  return shrunkenSeries;
}

} // anonymous namespace

template<typename T>
fastDTW<T>::fastDTW() {};

template<typename T>
fastDTW<T>::~fastDTW() {};

template<typename T>
warpInfo<T> fastDTW<T>::fullFastDTW(const std::vector<std::vector<T>>& seriesX, const std::vector<std::vector<T >>& seriesY, int searchRadius)
{

#ifndef EMSCRIPTEN
  if (seriesY.size() > seriesX.size())
  {
    return fullFastDTW(seriesY, seriesX, searchRadius); //TODO: I'm not sure why I need this. Also, not sure why it fails with Emscripten.
  }
#endif

  dtw<T> dtw;
  searchRadius = std::max(0, searchRadius);
  const size_t minSeries { static_cast<size_t>(searchRadius) + 2 };

  if (seriesX.size() <= minSeries || seriesY.size() <= minSeries)
  {
    return dtw.dynamicTimeWarp(seriesX, seriesY);
  }

  std::vector<std::vector<T>> shrunkenX { downsample(seriesX) };
  std::vector<std::vector<T>> shrunkenY { downsample(seriesY) };

  //some nice recursion here
  searchWindow window(seriesX.size(), seriesY.size(), getWarpPath(shrunkenX, shrunkenY, searchRadius), searchRadius);

  return dtw.constrainedDTW(seriesX, seriesY, window);
};

template<typename T>
T fastDTW<T>::getCost(const std::vector<std::vector<T>>& seriesX, const std::vector<std::vector<T>>& seriesY, int searchRadius)
{
  warpInfo<T> info { fullFastDTW(seriesX, seriesY, searchRadius) };
  return info.cost;
};

template<typename T>
warpPath fastDTW<T>::getWarpPath(const std::vector<std::vector<T>>& seriesX, const std::vector<std::vector<T>>& seriesY, int searchRadius)
{
  warpInfo<T> info { fullFastDTW(seriesX, seriesY, searchRadius) };
  return info.path;
};

//explicit instantiation
template class fastDTW<double>;
template class fastDTW<float>;
