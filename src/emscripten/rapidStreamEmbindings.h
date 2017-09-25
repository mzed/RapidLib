//
//  rapidStreamEmbindings.h
//  RapidLib
//
//  Created by mzed on 01/06/2017.
//  Copyright © 2017 Goldsmiths. All rights reserved.
//

#ifndef rapidStreamEmbindings_h
#define rapidStreamEmbindings_h

#include <emscripten/bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(rapidStream_module) {
  class_<rapidStream<double> >("RapidStreamCpp") //name change so that I can wrap it in Javascript. -mz
    .constructor()
    .constructor<int>()
    .function("clear", &rapidStream<double>::clear)
    .function("pushToWindow", &rapidStream<double>::pushToWindow)
    .function("velocity", &rapidStream<double>::velocity)
    .function("acceleration", &rapidStream<double>::acceleration)
    .function("minimum", &rapidStream<double>::minimum)
    .function("maximum", &rapidStream<double>::maximum)
    .function("sum", &rapidStream<double>::sum)
    .function("mean", &rapidStream<double>::mean)
    .function("standardDeviation", &rapidStream<double>::standardDeviation)
    .function("rms", &rapidStream<double>::rms)
    .function("minVelocity", &rapidStream<double>::minVelocity)
    .function("maxVelocity", &rapidStream<double>::maxVelocity)
    .function("minAcceleration", &rapidStream<double>::minAcceleration)
    .function("maxAcceleration", &rapidStream<double>::maxAcceleration)
    ;
  
};

#endif
