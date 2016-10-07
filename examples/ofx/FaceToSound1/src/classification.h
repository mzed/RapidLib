#ifndef classification_h
#define classification_h

#include <vector>
#include "modelSet.h"

/*! Class for implementing a set of classification models.
 *
 * This doesn't do anything modelSet can't do. But, it's simpler and more like wekinator.
 */

class classification : public modelSet {
public:
    /** with no arguments, just make an empty vector */
    classification();
    /** create based on training set inputs and outputs */
    classification(std::vector<trainingExample> trainingSet);
    /** create with proper models, but not trained */
    classification(int numInputs, int numOutputs);
    
    /** Train on a specified set, causes creation if not created */
    bool train(std::vector<trainingExample> trainingSet);
    
    std::vector<double> process(std::vector<double> inputVector);
};

#endif
