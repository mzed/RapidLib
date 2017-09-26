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
  class_<seriesClassification<double> >("SeriesClassificationCpp") //name change so that I can wrap it in Javascript. -mz
    .constructor()
    .function("reset", &seriesClassification<double>::reset)
    .function("train", &seriesClassification<double>::train)
    .function("run", &seriesClassification<double>::run)
    .function("getCosts", select_overload<std::vector<double>()>(&seriesClassification<double>::getCosts))
    ;
};

#endif
