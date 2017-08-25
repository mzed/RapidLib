#ifndef modelSetEmbindings_h
#define modelSetEmbindings_h

#include <emscripten/bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(modelSet_module) {
  class_<modelSet>("ModelSetCpp") //name change so that I can wrap it in Javascript. -mz
    .constructor()
    .function("train", &modelSet::train)
    .function("reset", &modelSet::reset)
    .function("run", &modelSet::run)
    ;
  
};

#endif
