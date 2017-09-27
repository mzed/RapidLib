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
    .function("run", select_overload<std::string(const std::vector<std::vector<double>>&)>(&seriesClassification<double>::run))
    .function("runLabel", select_overload<double(const std::vector<std::vector<double>>&, std::string)>(&seriesClassification<double>::run))
    .function("getCosts", &seriesClassification<double>::getCosts)
    ;
};

#endif
