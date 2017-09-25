#include <vector>
#include "classification.h"
#ifdef EMSCRIPTEN
#include "emscripten/classificationEmbindings.h"
#endif

template<typename T>
classification<T>::classification() {
    numInputs = 0;
    numOutputs = 0;
    created = false;
    classificationType = knn; //this is the default algorithm
};

template<typename T>
classification<T>::classification(classificationTypes classification_type) {
    numInputs = 0;
    numOutputs = 0;
    created = false;
    classificationType = classification_type;
};

template<typename T>
classification<T>::classification(const int &num_inputs, const int &num_outputs) { //TODO: this feature isn't really useful
    numInputs = num_inputs;
    numOutputs = num_outputs;
    created = false;
    std::vector<int> whichInputs;
    for (int i = 0; i < numInputs; ++i) {
        whichInputs.push_back(i);
    }
    std::vector<trainingExample<double> > trainingSet;
    for (int i = 0; i < numOutputs; ++i) {
        myModelSet.push_back(new knnClassification<double>(numInputs, whichInputs, trainingSet, 1));
    }
    created = true;
};

template<typename T>
classification<T>::classification(const std::vector<trainingExample<T> > &trainingSet) {
    numInputs = 0;
    numOutputs = 0;
    created = false;
    train(trainingSet);
};

template<typename T>
bool classification<T>::train(const std::vector<trainingExample<T> > &trainingSet) {
    //TODO: time this process?
    myModelSet.clear();
    //create model(s) here
    numInputs = int(trainingSet[0].input.size());
    for (int i = 0; i < numInputs; ++i) {
        inputNames.push_back("inputs-" + std::to_string(i + 1));
    }
    numOutputs = int(trainingSet[0].output.size());
    for ( auto example : trainingSet) {
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
        if (classificationType == svm) {
            myModelSet.push_back(new svmClassification<double>(numInputs));
        } else {
            myModelSet.push_back(new knnClassification<double>(numInputs, whichInputs, trainingSet, 1));
        }
    }
    created = true;
    return modelSet::train(trainingSet);
}

template<typename T>
std::vector<int> classification<T>::getK() {
    std::vector<int> kVector;
    for (baseModel<double>* model : myModelSet) {
        knnClassification<double>* kNNModel = dynamic_cast<knnClassification<double>*>(model); //FIXME: I really dislike this design
        kVector.push_back(kNNModel->getK());
    }
    return kVector;
}

template<typename T>
void classification<T>::setK(const int whichModel, const int newK) {
    knnClassification<double>* kNNModel = dynamic_cast<knnClassification<double>*>(myModelSet[whichModel]); //FIXME: I really dislike this design
    kNNModel->setK(newK);
}

//explicit instantiation
template class classification<double>;
//template class classification<float>;
