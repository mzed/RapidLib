//
//  classificationEmbindings.cpp
//  RapidLib
//
//  Created by mzed on 27/09/2016.
//  Copyright © 2016 Goldsmiths. All rights reserved.
//


#ifndef classificationEmbindings_h
#define classificationEmbindings_h

#include <emscripten/bind.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(classification_module) {
  class_<classification<double>, base<modelSet<double> > >("ClassificationCpp") //name change so that I can wrap it in Javascript. -mz
    .constructor()
    .constructor<classification<double>::classificationTypes>()
    //    .constructor< std::vector<trainingExample> >()
    .constructor<int, int>()
    .function("train", &classification<double>::train)
    .function("getK", &classification<double>::getK)
    .function("setK", &classification<double>::setK)
    ;
  enum_<classification<double>::classificationTypes>("ClassificationTypes")
    .value("KNN", classification<double>::knn)
    .value("SVM", classification<double>::svm)
    ;

};

#endif
