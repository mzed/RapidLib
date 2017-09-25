//
//  knnEmbindings.h
//  RapidLib
//
//  Created by mzed on 05/09/2016.
//  Copyright © 2016 Goldsmiths. All rights reserved.
//

#ifndef knnEmbindings_h
#define knnEmbindings_h

#include <vector>
#include <emscripten/bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(stl_wrappers) {
  register_vector<int>("VectorInt");
  register_vector<double>("VectorDouble");
  register_vector<std::vector<double>>("VectorVectorDouble");

  register_vector<trainingExample<double>>("TrainingSet");
  register_vector<trainingSeries<double>>("TrainingSeriesSet");

  value_object<trainingExample<double>>("trainingExample")
    .field("input", &trainingExample<double>::input)
    .field("output", &trainingExample<double>::output)
    ;

  value_object<trainingSeries<double>>("trainingSeries")
    .field("input", &trainingSeries<double>::input)
    .field("label", &trainingSeries<double>::label)
    ;
}


EMSCRIPTEN_BINDINGS(knn_module) {
  class_<knnClassification<double>>("KnnClassification")
    .constructor<int, std::vector<int>, std::vector<trainingExample<double>>, int>()
    .function("addNeighbour", &knnClassification<double>::addNeighbour)
    .function("run", &knnClassification<double>::run)
    ;
};

#endif
