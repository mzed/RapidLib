#include <vector>
#include "modelSet.h"

/** No arguments, don't create models yet */

modelSet::modelSet() :
numInputs(0),
numOutputs(0),
created(false)
{
};

modelSet::~modelSet() {
    for (std::vector<baseModel*>::iterator i = myModelSet.begin(); i != myModelSet.end(); ++i) {
        delete *i;
    }
};

bool modelSet::train(std::vector<trainingExample> training_set) {
    for (int i = 0; i < myModelSet.size(); ++i) {
        std::vector<trainingExample> modelTrainingSet; //just one output
        for (auto example : training_set) {
            std::vector<double> tempDouble;
            for (int j = 0; j < numInputs; ++j) {
                tempDouble.push_back(example.input[j]);
            }
            trainingExample tempObj = {tempDouble, std::vector<double> {example.output[i]}};
            modelTrainingSet.push_back(tempObj);
        }
        myModelSet[i]->train(modelTrainingSet);
    }
    created = true;
    return true;
}

std::vector<double> modelSet::process(std::vector<double> inputVector) {
  std::vector<double> returnVector;
    if (created) {
        for (auto model : myModelSet) {
	  returnVector.push_back(model->process(inputVector));
        }
    } else {
        returnVector.push_back(0);
    }
    return returnVector;
}