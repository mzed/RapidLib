/**
 * @file  warpPath.h
 * RapidLib
 * @author Michael Zbyszynski
 * @date 13 Sep 2017
 * @copyright Copyright © 2017 Goldsmiths. All rights reserved.
 */

#pragma once

#include <vector>

/** Struct to hold warp paths through a cost matrix */
struct warpPath
{
  std::vector<std::pair<std::size_t, std::size_t>> indices{};

  /**
   * Add a point to the beginning of the warp path.
   * (Paths are calculated starting from the end)
   */

  void add(const std::size_t x, const std::size_t y)
  {
    indices.insert(indices.begin(), std::make_pair(x, y));
  };
};

/** return struct holding a warp path and the cost of that path */
template <typename T> struct warpInfo
{
  warpPath path{};
  T cost{};
};
