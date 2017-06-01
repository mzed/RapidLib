# Link to [C++ documentation](http://doc.gold.ac.uk/eavi/rapidmix/docs_cpp/annotated.html)


# JavaScript documentation:





* * *

### prepTrainingSet(trainingSet) 

Utility function to convert js objects into something emscripten likes

**Parameters**

**trainingSet**: `Object`, JS Object representing a training set

**Returns**: `Module.TrainingSet`


## Class: Regression
Creates a set of regression objects using the constructor from emscripten

**Module.RegressionCpp**: `function` , constructor from emscripten
### Regression.train(trainingSet) 

Trains the models using the input. Starts training from the current state of the model: randomized or trained.

**Parameters**

**trainingSet**: `Object`, An array of training examples

**Returns**: `Boolean`, true indicates successful training

### Regression.initialize() 

Returns the model set to it's initial configuration.

**Returns**: `Boolean`, true indicates successful initialization

### Regression.process(input) 

Runs feed-forward regression on input

**Parameters**

**input**: `Array`, An array of features to be processed. Non-arrays are converted.

**Returns**: `Array`, output - One number for each model in the set


## Class: Classification
Creates a set of classification objects using the constructor from emscripten

**Module.ClassificationCpp**: `function` , constructor from emscripten
### Classification.train(trainingSet) 

Trains the models using the input. Clears previous training set.

**Parameters**

**trainingSet**: `Object`, An array of training examples.

**Returns**: `Boolean`, true indicates successful training

### Classification.initialize() 

Returns the model set to it's initial configuration.

**Returns**: `Boolean`, true indicates successful initialization

### Classification.process(input) 

Does classifications on an input vector.

**Parameters**

**input**: `Array`, An array of features to be processed. Non-arrays are converted.

**Returns**: `Array`, output - One number for each model in the set


## Class: ModelSet
Creates a set of machine learning objects using constructors from emscripten. Could be any mix of regression and classification.

### ModelSet.loadJSON(url) 

Trains the models using the input. Clears previous training set.

**Parameters**

**url**: `string`, JSON loaded from a model set description document.

**Returns**: `Boolean`, true indicates successful training

### ModelSet.addNNModel(model) 

Add a NN model to a modelSet. //TODO: this doesn't need it's own function

**Parameters**

**model**: , Add a NN model to a modelSet. //TODO: this doesn't need it's own function


### ModelSet.addkNNModel(model) 

Add a kNN model to a modelSet. //TODO: this doesn't need it's own function

**Parameters**

**model**: , Add a kNN model to a modelSet. //TODO: this doesn't need it's own function


### ModelSet.process(input) 

Applies regression and classification algorithms to an input vector.

**Parameters**

**input**: `Array`, An array of features to be processed.

**Returns**: `Array`, output - One number for each model in the set



* * *










