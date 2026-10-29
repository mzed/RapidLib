/**
 * @file    searchWindow.h
 * RapidLib
 * @author  Michael Zbyszynski
 * @date    14 Sep 2017
 * @copyright Copyright © 2017 Goldsmiths. All rights reserved.
 */

#pragma once

#include "warpPath.h"
#include <vector>

/** For fastDTW, a window of cells to calculate costs for.
 * These are cells within a specified search radius of a lower resolution path.
 */
class searchWindow
{
public:
  searchWindow(const std::size_t seriesXSize,
               const std::size_t seriesYSize,
               const warpPath &shrunkenWarpPath,
               const int searchRadius);

  std::vector<std::pair<int, int>> minMaxValues {};

private:
  inline void markVisited(int col, int row);
  void expandWindow(int searchRadius);

  int maxY {};
};
