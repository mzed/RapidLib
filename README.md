Source files for generating RapidMix Javascript library.

# Rapid API

## Basic objects

`var myThing = new RapidLib.Regression();`  
This creates a set of regression models using neural networks. The models will have one hidden layer with the same number of nodes as there are inputs in the training set (see below).  There is one model per output.

`var myThing = new RapidLib.Classification();`  
This creates a set of classification models using k nearest neighbour classification. k always = 1.

### _optional arguments_
These constructors can be specified with arguments for numbers of inputs and outputs, like:  
`var myOptionalThing = new RapidLib.Regression(9, 4)`  
This regression object expects nine inputs and returns 4 outputs.

Also, a training set can be specificed, like:  
`var myTrainedClassifier = new Rapid.Classification(trainingSet);`  
See below for training set format.

## Training
Newly created machine learning objects need to be trained before they will give proper output. To do so, they are fed a set of training data. A training set is an array of training examples in the following form:

`var myTrainingSet = [`  
`{`  
`   input: [0, 1, 2],`  
`   output: [0, 1]`  
`}`  

A live example is [here]. (http://live.codecircle.com/d/wiCgiE7ogQXFgMEMt)

This is passed to the machine learning object using the `train()` method.

`var myReg = new RapidLib.Regression();`  
`myReg.train(trainingSet);`  

Returns `true` when models are trained.

## Running
Trained models process input using the `process` function. Input is an array of numbers (doubles).

`var myInput = [90.0, 3.14, 1.618];`  
`var modelOutput = myReg.procss(myInput);`  

The model ouput will be an array of numbers (doubles).
