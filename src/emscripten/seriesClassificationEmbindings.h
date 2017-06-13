#ifndef seriesClassificationEmbindings_h
#define seriesClassificationEmbindings_h

#include <emscripten.h>
#include <bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(seriesClassification_module) {
  class_<seriesClassification>("SeriesClassificationCpp") //name change so that I can wrap it in Javascript. -mz
    .constructor()
    .function("addTrainingSet", &seriesClassification::addTrainingSet)
    .function("train", &seriesClassification::train)
    .function("reset", &seriesClassification::reset)
    .function("runTrainingSet", &seriesClassification::runTrainingSet)
    ;

};

#endif
