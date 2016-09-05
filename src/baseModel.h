#ifndef baseModel_h
#define baseModel_h

#include <vector>

class baseModel {
public:
  virtual double processInput(std::vector<double>) = 0;
    virtual ~baseModel() {};
};

#endif
