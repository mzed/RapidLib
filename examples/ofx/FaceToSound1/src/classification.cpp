#include <vector>
#include "classification.h"
//#include "classificationEmbindings.h"

classification::classification() {
    numInputs = 0;
    numOutputs = 0;
    created = false;
};

classification::classification(int num_inputs, int num_outputs) {
    numInputs = 0;
    numOutputs = 0;
    created = false;
    std::vector<int> whichInputs;
    for (int i = 0; i < numInputs; ++i) {
        whichInputs.push_back(i);
    }
    std::vector<trainingExample> trainingSet;
    for (int i = 0; i < numOutputs; ++i) {
        myModelSet.push_back(new knnClassification(numInputs, whichInputs, trainingSet, 1));
    }
    created = true;
};

classification::classification(std::vector<trainingExample> training_set) {
    numInputs = 0;
    numOutputs = 0;
    created = false;
    train(training_set);
};

bool classification::train(std::vector<trainingExample> training_set) {
    //TODO: time this process?
    if (created) {
      return modelSet::train(training_set);
    } else {
        //create model(s) here
        numInputs = training_set[0].input.size();
        numOutputs = training_set[0].output.size();
        for ( auto example : training_set) {
            if (example.input.size() != numInputs) {
                return false;
            }
            if (example.output.size() != numOutputs) {
                return false;
            }
        }
        std::vector<int> whichInputs;
        for (int j = 0; j < numInputs; ++j) {
            whichInputs.push_back(j);
        }
        for (int i = 0; i < numOutputs; ++i) {
            myModelSet.push_back(new knnClassification(numInputs, whichInputs, training_set, 1));
        }
        created = true;
        return modelSet::train(training_set);
    }
}

std::vector<double> classification::process(std::vector<double> inputVector) {
    //Emscripten made me do it. -mz
    return modelSet::process(inputVector);
}
