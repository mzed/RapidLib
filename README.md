Source files for generating RapidMix Javascript library.

# Rapid API

###Basic objects

`var myThing = new RapidLib.Regression();` 
This creates a set of regression models using neural networks. The models will have one hidden layer with the same number of nodes as there are inputs in the training set (see below).  There is one model per output.

`var myThing = new RapidLib.Classification();'
This creates a set of classification models using k nearest neighbour classification. k always = 1.

_optional arguments_
These constructors can be specified with arguments for numbers of inputs and outputs, like:
`var myOptionalThing = new RapidLib.Regression(9, 4)`
This regression object expects nine inputs and returns 4 outputs.

Also, a training set can be specificed, like:
`var myTrainedClassifier = new Rapid.Classification(trainingSet);`
See below for training set format.

