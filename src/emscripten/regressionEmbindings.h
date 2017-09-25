//
//  regressionEmbindings.cpp
//  RapidLib
//
//  Created by mzed on 26/09/2016.
//  Copyright © 2016 Goldsmiths. All rights reserved.
//

#ifndef regressionEmbindings_h
#define regressionEmbindings_h

#include <emscripten/bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(regression_module) {
  class_<regression<double>, base<modelSet<double> > >("RegressionCpp") //name change so that I can wrap it in Javascript. -mz
    .constructor()
    .constructor< std::vector<trainingExample<double> > >()
    .constructor<int, int>()
    .function("train", &regression<double>::train)
    .function("getNumHiddenLayers", &regression<double>::getNumHiddenLayers)
    .function("setNumHiddenLayers", &regression<double>::setNumHiddenLayers)
    .function("setNumEpochs", &regression<double>::setNumEpochs)
    ;

};

#endif
