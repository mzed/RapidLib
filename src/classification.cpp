#include <vector>
#include "classification.h"
#ifdef EMSCRIPTEN
#include "classificationEmbindings.h"
#endif

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
        numInputs = int(training_set[0].input.size());
        for (int i = 0; i < numInputs; ++i) {
            inputNames.push_back("input-" + std::to_string(i));
        }
        numOutputs = int(training_set[0].output.size());
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
