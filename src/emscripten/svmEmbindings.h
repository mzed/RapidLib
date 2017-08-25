#ifndef svmEmbindings_h
#define svmEmbindings_h

#include <vector>
#include <emscripten/bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(svm_module) {
  class_<svmClassification>("svmClassificationCPP")
    .constructor<int>()
    .function("train", &svmClassification::train)
    .function("run", &svmClassification::run)
    ;
};

#endif
