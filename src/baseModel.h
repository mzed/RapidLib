#ifndef baseModel_h
#define baseModel_h

#include <vector>

#ifndef EMSCRIPTEN
#include "json.h"
#endif

/** This is used by both NN and KNN models for training and classification */
struct trainingExample {
    std::vector<double> input;
    std::vector<double> output;
};

/** Base class for wekinator models. Implemented by NN and KNN classes
 */
class baseModel {
public:
    virtual ~baseModel() {};
    virtual double process(std::vector<double>) = 0;
    virtual void train(std::vector<trainingExample>) = 0;
    virtual int getNumInputs() = 0;
    virtual std::vector<int> getWhichInputs() = 0;
#ifndef EMSCRIPTEN
    virtual void getJSONDescription(Json::Value &currentModel) = 0;
    
protected:
    template<typename T>
    Json::Value vector2json(T vec) {
        Json::Value toReturn;
        for (int i = 0; i < vec.size(); ++i) {
            toReturn.append(vec[i]);
        }
        return toReturn;
    }
    
#endif
    
};
#endif
