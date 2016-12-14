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
    classification(const std::vector<trainingExample> &trainingSet);
    /** create with proper models, but not trained */
    classification(const int &numInputs, const int &numOutputs);
    
    /** destructor */
    ~classification() {}
    
    /** Train on a specified set, causes creation if not created */
    bool train(const std::vector<trainingExample> &trainingSet);
    
    /** Check the K values for each model. This feature is temporary, and will be replaced by a different design. */
    std::vector<int> getK();
    /** Get the K values for each model. This feature is temporary, and will be replaced by a different design. */
    void setK(const int whichModel, const int newK);
};

#endif
