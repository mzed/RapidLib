#include <vector>
#include "classification.h"
#ifdef EMSCRIPTEN
#include "emscripten/classificationEmbindings.h"
#endif

classification::classification() {
    numInputs = 0;
    numOutputs = 0;
    created = false;
};

classification::classification(const int &num_inputs, const int &num_outputs) { //TODO: this feature isn't really useful
    numInputs = num_inputs;
    numOutputs = num_outputs;
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

classification::classification(const std::vector<trainingExample> &training_set) {
    numInputs = 0;
    numOutputs = 0;
    created = false;
    train(training_set);
};

bool classification::train(const std::vector<trainingExample> &training_set) {
    //TODO: time this process?
    if (created) {
      return modelSet::train(training_set);
    } else {
        //create model(s) here
        numInputs = int(training_set[0].input.size());
        for (int i = 0; i < numInputs; ++i) {
            inputNames.push_back("inputs-" + std::to_string(i + 1));
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

std::vector<int> classification::getK() {
    std::vector<int> kVector;
    for (baseModel* model : myModelSet) {
        knnClassification* kNNModel = dynamic_cast<knnClassification*>(model); //FIXME: I really dislike this design
        kVector.push_back(kNNModel->getK());
    }
    return kVector;
}

void classification::setK(const int whichModel, const int newK) {
        knnClassification* kNNModel = dynamic_cast<knnClassification*>(myModelSet[whichModel]); //FIXME: I really dislike this design
        kNNModel->setK(newK);
}