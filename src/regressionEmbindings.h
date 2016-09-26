#ifndef regressionEmbindings_h
#define regressionEmbindings_h

#include <emscripten.h>
#include <bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(regression_module) {
  //  class_<modelSet>("modelSet");
  class_<regression>("Regression")
    .constructor()
    .constructor< std::vector<trainingExample> >()
    .constructor<int, int>()
    .function("train", &regression::train)
    .function("process", &regression::process)
    ;

};

#endif
