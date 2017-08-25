#ifndef seriesClassificationEmbindings_h
#define seriesClassificationEmbindings_h

#include <emscripten/bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(seriesClassification_module) {
  class_<seriesClassification>("SeriesClassificationCpp") //name change so that I can wrap it in Javascript. -mz
    .constructor()
    //.function("addTrainingSet", &seriesClassification::addTrainingSet)
    //.function("train", &seriesClassification::train)
    .function("reset", &seriesClassification::reset)
    .function("trainLabel", &seriesClassification::trainLabel)
    .function("runLabel", &seriesClassification::runLabel)
    //.function("runTrainingSet", &seriesClassification::runTrainingSet)
    .function("getCosts", select_overload<std::vector<double>()>(&seriesClassification::getCosts))
    ;

};

#endif
