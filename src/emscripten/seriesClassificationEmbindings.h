//
//  seriesClassification.h
//  RapidLib
//
//  Created by mzed on 13/06/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//


#ifndef seriesClassificationEmbindings_h
#define seriesClassificationEmbindings_h

#include <emscripten/bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(seriesClassification_module) {
  class_<seriesClassification>("SeriesClassificationCpp") //name change so that I can wrap it in Javascript. -mz
    .constructor()
    .function("reset", &seriesClassification::reset)
    .function("train", &seriesClassification::train)
    .function("run", &seriesClassification::run)
    .function("getCosts", select_overload<std::vector<double>()>(&seriesClassification::getCosts))
    ;
};

#endif
