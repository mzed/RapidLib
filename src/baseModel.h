#ifndef baseModel_h
#define baseModel_h

#include <vector>

struct trainingExample {
    std::vector<double> input;
    double output;
};

class baseModel {
public:
    virtual double process(std::vector<double>) = 0;
    virtual void train(std::vector<trainingExample>) = 0;
    virtual ~baseModel() {};
};

#endif
