#ifndef knnEmbindings_h
#define knnEmbindings_h

#include <vector>
#include <emscripten.h>
#include <bind.h>

using namespace emscripten;


EMSCRIPTEN_BINDINGS(stl_wrappers) {
  register_vector<int>("VectorInt");
  register_vector<double>("VectorDouble");
  register_vector<std::vector<double>>("VectorVectorDouble");

  register_vector<trainingExample>("TrainingSet");
  register_vector<trainingSeries>("TrainingSeriesSet");

  value_object<trainingExample>("trainingExample")
    .field("input", &trainingExample::input)
    .field("output", &trainingExample::output)
    ;

  value_object<trainingSeries>("trainingSeries")
    .field("input", &trainingSeries::input)
    .field("label", &trainingSeries::label)
    ;
}


EMSCRIPTEN_BINDINGS(knn_module) {
  class_<knnClassification>("KnnClassification")
    .constructor<int, std::vector<int>, std::vector<trainingExample>, int>()
    .function("addNeighbour", &knnClassification::addNeighbour)
    .function("run", &knnClassification::run)
    ;
};

#endif
