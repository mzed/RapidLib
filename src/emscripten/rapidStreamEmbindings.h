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
  class_<rapidStream>("RapidStreamCpp") //name change so that I can wrap it in Javascript. -mz
    .constructor()
    .constructor<int>()
    .function("clear", &rapidStream::clear)
    .function("pushToWindow", &rapidStream::pushToWindow)
    .function("velocity", &rapidStream::velocity)
    .function("acceleration", &rapidStream::acceleration)
    .function("minimum", &rapidStream::minimum)
    .function("maximum", &rapidStream::maximum)
    .function("sum", &rapidStream::sum)
    .function("mean", &rapidStream::mean)
    .function("standardDeviation", &rapidStream::standardDeviation)
    .function("rms", &rapidStream::rms)
    .function("minVelocity", &rapidStream::minVelocity)
    .function("maxVelocity", &rapidStream::maxVelocity)
    .function("minAcceleration", &rapidStream::minAcceleration)
    .function("maxAcceleration", &rapidStream::maxAcceleration)
    ;
  
};

#endif
