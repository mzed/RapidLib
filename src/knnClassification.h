#ifndef knnClassification_h
#define knnClassification_h

#include <vector>
#include "baseModel.h"

#ifndef EMSCRIPTEN
#include "json.h"
#endif

/** Class for implementing a knn classifier */
class knnClassification : public baseModel {
    
public:
    /** Constructor that takes training examples in
     * @param number of inputs expected in the training and input vectors
     * @param vector of input numbers to be fed into the classifer.
     * @param vector of training examples
     * @param how many near neighbours to evaluate
     */
    knnClassification(int num_inputs, std::vector<int> which_inputs, std::vector<trainingExample> trainingSet, int k);
    ~knnClassification();
    
    /** add another example to the existing training set
     * @param class number of example
     * @param feature vector of example
     */
    void addNeighbour(int, std::vector<double>);
    
    /** Generate an output value from a single input vector.
     * @param A standard vector of doubles to be evaluated.
     * @return A single double: the nearest class as determined by k-nearest neighbor.
     */
    double process(const std::vector<double> &inputVector);
    
    /** Fill the model with a vector of examples.
     *
     * @param The training set is a vector of training examples that contain both a vector of input values and a double specifying desired output class.
     *
     */
    void train(const std::vector<trainingExample> &trainingSet);
    
    int getNumInputs();
    std::vector<int> getWhichInputs();
    
#ifndef EMSCRIPTEN
    void getJSONDescription(Json::Value &currentModel);
#endif
    
private:
    int numInputs;
    std::vector<int> whichInputs;
    std::vector<trainingExample> neighbours;
    int numNeighbours; //aka "k"
    std::pair<int, double>* nearestNeighbours;
};

#endif

