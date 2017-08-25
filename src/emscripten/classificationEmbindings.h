#ifndef classificationEmbindings_h
#define classificationEmbindings_h

#include <emscripten/bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(classification_module) {
  class_<classification, base<modelSet>>("ClassificationCpp") //name change so that I can wrap it in Javascript. -mz
    .constructor()
    .constructor<classification::classificationTypes>()
    //    .constructor< std::vector<trainingExample> >()
    .constructor<int, int>()
    .function("train", &classification::train)
    .function("getK", &classification::getK)
    .function("setK", &classification::setK)
    ;
  enum_<classification::classificationTypes>("ClassificationTypes")
    .value("KNN", classification::knn)
    .value("SVM", classification::svm)
    ;

};

#endif
