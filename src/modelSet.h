/**
 * @file modelSet.h
 * RapidLib
 *
 * @author Michael Zbyszynski
 * @date 26 Sep 2016
 * @copyright Copyright © 2016 Goldsmiths. All rights reserved.
 */


#ifndef MODELSET_H
#define MODELSET_H

#include <vector>
#include "trainingExample.h"
#include "baseModel.h"
#include "neuralNetwork.h"
#include "knnClassification.h"
#include "svmClassification.h"
#ifndef EMSCRIPTEN
#include "../dependencies/json/json.h"
#endif

#include "rapidlib_export.h"

/** This class holds a set of models with the same or different algorithms. */
template<typename T>
class modelSet {
public:
    RAPIDLIB_EXPORT modelSet();
    RAPIDLIB_EXPORT virtual ~modelSet();
    /** Train on a specified set, causes creation if not created */
    RAPIDLIB_EXPORT virtual bool train(const std::vector<trainingExampleTemplate<T> > &trainingSet);
    /** reset to pre-training state */
    RAPIDLIB_EXPORT bool reset();

    /** Generate an output value from a single input vector.
    *
    * Will return an error if training in progress.
    * 
    * @param vector A standard vector of type T that is the input for classification or regression.
    * @return vector A vector type T that are the predictions for each model in the set.
    */
    RAPIDLIB_EXPORT std::vector<T> run(const std::vector<T> &inputVector);

protected:
    std::vector<baseModel<T>*> models {};
    int numInputs {};
    std::vector<std::string> inputNames {};
    int numOutputs {};
    bool isTraining {}; //This is true while the models are training, and will block running
    bool isTrained {};

    void threadTrain(std::size_t i, const std::vector<trainingExampleTemplate<T> >& training_set);

#ifndef EMSCRIPTEN //The javascript code will do its own JSON parsing
public:

    /** Get a JSON representation of the model
    * 
    * @return Styled string JSON representation
    */
    RAPIDLIB_EXPORT std::string getJSON();

    /** Write a JSON model description to specified file path 
    *
    * @param file path
    * 
    */
    RAPIDLIB_EXPORT void writeJSON(const std::string &filepath);

    /** configure empty model with string. See getJSON() */
    RAPIDLIB_EXPORT bool putJSON(const std::string &jsonMessage);

    /** read a JSON file at file path and build a modelSet from it */
    RAPIDLIB_EXPORT bool readJSON(const std::string &filepath);

private:
    Json::Value parse2json();
    void json2modelSet(const Json::Value &root);
#endif
};

#endif
