/*
bindings for use with emscripten. -22 Aug 2016, mz
*/

#ifndef knnEmbindings_h
#define knnEmbindings_h

#include <vector>
#include <emscripten.h>
#include <bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(stl_wrappers) {
  register_vector<int>("VectorInt");
  register_vector<double>("VectorDouble");
}


EMSCRIPTEN_BINDINGS(knn_module) {
  register_vector<neighbour>("VectorNeighbour");

  value_object<neighbour>("neighbour")
    .field("classNum", &neighbour::classNum)
    .field("featurs", &neighbour::features)
    ;

  class_<knnClassification>("knnClassification")
    .constructor<int, std::vector<int>, std::vector<neighbour>, int, int, int>()
    .function("addNeighbour", &knnClassification::addNeighbour)
    .function("processInput", &knnClassification::processInput)
    ;
  };
#endif
