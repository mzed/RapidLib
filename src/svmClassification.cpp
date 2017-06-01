#include <iostream>
#include "svmClassification.h"

svmClassification::svmClassification(
         KernelType kernelType,
         SVMType svmType,
         bool useScaling,
         bool useNullRejection,
         bool useAutoGamma,
         float gamma,
         unsigned int degree,
         float coef0,
         float nu,
         float C,
         bool useCrossValidation,
         unsigned int kFoldValue
         )
{
    
    //Setup the default SVM parameters
    model = NULL;
    param.weight_label = NULL;
    param.weight = NULL;
    prob.l = 0;
    prob.x = NULL;
    prob.y = NULL;
    trained = false;
    problemSet = false;
    param.svm_type = C_SVC;
    param.kernel_type = LINEAR_KERNEL;
    param.degree = 3;
    param.gamma = 0;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 1;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    
    //These are from GTK?
    /*
     this->useScaling = false;
     this->useCrossValidation = false;
     this->useNullRejection = false;
     this->useAutoGamma = true;
     classificationThreshold = 0.5;
     crossValidationResult = 0;
     
     classifierMode = STANDARD_CLASSIFIER_MODE;
     */
    
    init(kernelType,svmType,useScaling,useNullRejection,useAutoGamma,gamma,degree,coef0,nu,C,useCrossValidation,kFoldValue);
}

svmClassification::~svmClassification() {
    
}

bool svmClassification::init(
               KernelType kernelType,
               SVMType svmType,
               bool useScaling,
               bool useNullRejection,
               bool useAutoGamma,
               float gamma,
               unsigned int degree,
               float coef0,
               float nu,
               float C,
               bool useCrossValidation,
               unsigned int kFoldValue
               ){
    
    /*
    //Clear any previous models or problems
    clear();
    
    //Validate the kernerlType
    if( !validateKernelType(kernelType) ){
        errorLog << __GRT_LOG__ << " Unknown kernelType!\n";
        return false;
    }
    
    if( !validateSVMType(svmType) ){
        errorLog << __GRT_LOG__ << " Unknown kernelType!\n";
        return false;
    }
    */
    
    param.svm_type = (int)svmType;
    param.kernel_type = (int)kernelType;
    param.degree = (int)degree;
    param.gamma = gamma;
    param.coef0 = coef0;
    param.nu = nu;
    param.cache_size = 100;
    param.C = C;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 1;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    /*
    this->useScaling = useScaling;
    this->useCrossValidation = useCrossValidation;
    this->useNullRejection = useNullRejection;
    this->useAutoGamma = useAutoGamma;
    classificationThreshold = 0.5;
    crossValidationResult = 0;
    */
    
    return true;
}

void svmClassification::train(const std::vector<trainingExample> &trainingSet) {
    std::cout << "SVM could be training" << std::endl;
};

double svmClassification::process(const std::vector<double> &inputVector) {
    return 0;
}

int svmClassification::getNumInputs() const {
    return 0;
};

std::vector<int> svmClassification::getWhichInputs() const {
    std::vector<int> returnVec;
    return returnVec;
};

void svmClassification::getJSONDescription(Json::Value &currentModel){
    
};
