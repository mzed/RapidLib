#include <vector>
#include "regression.h"
//#include "regressionEmbindings.h"

regression::regression() {
    numInputs = 0;
    numOutputs = 0;
    created = false;
};

regression::regression(int num_inputs, int num_outputs) {
    numInputs = 0;
    numOutputs = 0;
    created = false;
    std::vector<int> whichInputs;
    for (int i = 0; i < numInputs; ++i) {
        whichInputs.push_back(i);
    }
    for (int i = 0; i < numOutputs; ++i) {
        myModelSet.push_back(new neuralNetwork(numInputs, whichInputs, 1, numInputs));
    }
    created = true;
};

regression::regression(std::vector<trainingExample> training_set) {
    numInputs = 0;
    numOutputs = 0;
    created = false;
    train(training_set);
};

bool regression::train(std::vector<trainingExample> training_set) {
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
            myModelSet.push_back(new neuralNetwork(numInputs, whichInputs, 1, numInputs));
	}
        created = true;
        return modelSet::train(training_set);
    }
}

std::vector<double> regression::process(std::vector<double> inputVector) {
  //Emscripten made me do it. -mz
  return modelSet::process(inputVector);
}
