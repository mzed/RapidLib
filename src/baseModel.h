#ifndef baseModel_h
#define baseModel_h

#include <vector>

/** This is used by both NN and KNN models for training and classification */
struct trainingExample {
    std::vector<double> input;
    double output;
};

/** Base class for wekinator models. Implemented by NN and KNN classes
 */
class baseModel {
public:
    virtual double process(std::vector<double>) = 0;
    virtual void train(std::vector<trainingExample>) = 0;
    virtual ~baseModel() {};
};

#endif
