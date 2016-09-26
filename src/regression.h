#ifndef regression_h
#define regression_h

#include <vector>
#include "modelSet.h"

/*! Class for implementing a set of regression models.
 *
 * This doesn't do anything modelSet can't do. But, it's simpler and more like wekinator.
 */

class regression : public modelSet {
public:
    /** with no arguments, just make an empty vector */
    regression();
    /** create based on training set inputs and outputs */
    regression(std::vector<trainingExample> trainingSet);
    /** create with proper models, but not trained */
    regression(int numInputs, int numOutputs);
    
    /** Train on a specified set, causes creation if not created */
    bool train(std::vector<trainingExample> trainingSet);    

    std::vector<double> process(std::vector<double> inputVector); 
};

#endif
