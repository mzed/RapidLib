/* globals Module */

"use strict";

Module.prepTrainingSet = function (trainingSet) {
    var rmTrainingSet = new rapidMix.TrainingSet();
    for (var i = 0; i < trainingSet.length; ++i) {
        var tempInput = new rapidMix.VectorDouble();
        var tempOutput = new rapidMix.VectorDouble();
        for (var j = 0; j < trainingSet[i].input.length; ++j) {
            tempInput.push_back(parseFloat(trainingSet[i].input[j]));
        }
        for (var j = 0; j < trainingSet[i].output.length; ++j) {
            tempOutput.push_back(parseFloat(trainingSet[i].output[j]));
        }
        var tempObj = {'input': tempInput, 'output': tempOutput};
        rmTrainingSet.push_back(tempObj);
     }
    return rmTrainingSet;
};


/*
 Module.Regression = function () {
 this.model = [];
 this.numInputs = 0;
 this.numOutputs = 0;
 this.created = false;
 switch (arguments.length) {
 case 0:
 break;
 case 1:
 this.train(arguments[0]);
 break;
 case 2:
 this.numInputs = arguments[0];
 this.numOutputs = arguments[1];
 var whichInputs = new Module.VectorInt;
 for (var i = 0; i < this.numInputs; ++i) {
 whichInputs.push_back(i);
 }
 for (var i = 0; i < this.numOutputs; ++i) {
 this.model.push(new Module.NeuralNetwork(this.numInputs, whichInputs, 1, this.numInputs));
 }
 this.created = true;
 break;
 default:
 console.error('rapidMix regression takes 2 arguments: # of inputs, # of outputs');
 }
 };
 */

/*
 Module.Regression.prototype = {
 train: function (trainingSet) {
 var start = new Date().getTime();
 if (this.created) {
 for (var i = 0; i < this.model.length; ++i) {
 var rmTrainingSet = new Module.TrainingSet;
 for (var ex in trainingSet) {
 var tempDouble = new Module.VectorDouble;
 for (var j = 0; j < this.numInputs; ++j) {
 tempDouble.push_back(parseFloat(trainingSet[ex].input[j]));
 }
 var tempObj = {'input': tempDouble, 'output': parseFloat(trainingSet[ex].output[i])};
 rmTrainingSet.push_back(tempObj);
 }
 this.model[i].train(rmTrainingSet);
 var end = new Date().getTime();
 var time = end - start;
 var result = {'examples': trainingSet.length, 'time': time};
 console.log(result);
 }
 return true;
 } else {
 ///create model(s) here
 this.numInputs = trainingSet[0].input.length;
 this.numOutputs = trainingSet[0].output.length;
 for (var ex in trainingSet) {
 if (trainingSet[ex].input.length != this.numInputs) {
 console.error("training set examples have different numbers of inputs");
 return false;
 }
 if (trainingSet[ex].output.length != this.numOutputs) {
 console.error("training set examples have different numbers of outputs");
 return false;
 }
 }

 var whichInputs = new Module.VectorInt;
 for (var j = 0; j < this.numInputs; ++j) {
 whichInputs.push_back(j);
 }
 for (var i = 0; i < this.numOutputs; ++i) {
 this.model.push(new Module.NeuralNetwork(this.numInputs, whichInputs, 1, this.numInputs))
 }
 this.created = true;
 return this.train(trainingSet);
 }
 },
 process: function (input) {
 if (this.created) {
 var returnArray = [];
 var inputVector = new Module.VectorDouble;
 for (var j = 0; j < input.length; ++j) {
 inputVector.push_back(input[j]);
 }

 for (var i = 0; i < this.model.length; ++i) {
 returnArray.push(this.model[i].process(inputVector));
 }
 return returnArray;
 } else {
 console.error("No trained model here");
 return false;
 }
 }
 };
 */

Module.Classification = function () {
    this.model = [];
    this.numInputs = 0;
    this.numOutputs = 0;
    this.created = false;
    switch (arguments.length) {
        case 0:
            break;
        case 2:
            var trainingSet = new Module.TrainingSet;
            this.numInputs = arguments[0];
            var whichInputs = new Module.VectorInt;
            for (var i = 0; i < this.numInputs; ++i) {
                whichInputs.push_back(i);
            }
            this.numOutputs = arguments[1];
            this.model = [];
            for (var i = 0; i < this.numOutputs; ++i) {
                this.model.push(new Module.KnnClassification(this.numInputs, whichInputs, trainingSet, 1));
            }
            this.created = true;
            break;
        default:
            console.error('rapidMix classification takes 2 arguments: # of inputs, # of outputs');
    }
};

Module.Classification.prototype = {
    train: function (trainingSet) {
        var start = new Date().getTime();
        if (this.created) {
            for (var out = 0; out < this.numOutputs; ++out) {
                for (var i = 0; i < trainingSet.length; ++i) {
                    var features = new Module.VectorDouble;
                    for (var j = 0; j < this.numInputs; ++j) {
                        features.push_back(trainingSet[i].input[j]);
                    }
                    this.model[out].addNeighbour(trainingSet[i].output[out], features);
                }
                var end = new Date().getTime();
                var time = end - start;
                var result = {'examples': trainingSet.length, 'time': time};
                console.log(result);
            }
            return true;
        } else {
            this.numInputs = trainingSet[0].input.length;
            this.numOutputs = trainingSet[0].output.length;

            var rmTrainingSet = new Module.TrainingSet();
            for (var ex in trainingSet) {
                if (trainingSet[ex].input.length != this.numInputs) {
                    console.error("training set examples have different numbers of inputs");
                    return false;
                }
                if (trainingSet[ex].output.length != this.numOutputs) {
                    console.error("training set examples have different numbers of outputs");
                    return false;
                }
            }
            var whichInputs = new Module.VectorInt;
            for (var i = 0; i < this.numInputs; ++i) {
                whichInputs.push_back(i);
            }
            for (var output = 0; output < this.numOutputs; ++output) {
                this.model.push(new Module.KnnClassification(this.numInputs, whichInputs, rmTrainingSet, 1));
            }
            this.created = true;
            this.train(trainingSet);
        }
    },
    process: function (input) {
        if (this.created) {
            var toKnn = new Module.VectorDouble();
            for (var i = 0; i < this.numInputs; ++i) {
                toKnn.push_back(input[i]);
            }
            var returnArray = [];
            for (var i = 0; i < this.model.length; ++i) {
                returnArray.push(this.model[i].process(toKnn));
            }
            return returnArray;
        } else {
            console.error("No trained model here");
            return false;
        }
    }
};

Module.ModelSet = function () {
    console.log("creating model set");
    this.myModelSet = [];
};

Module.ModelSet.prototype.loadJSON = function (url) {
    var that = this;
    var request = new XMLHttpRequest();
    request.open("GET", url, true);
    request.responseType = "json";
    request.onload = function () {
        //console.log("loaded", JSON.stringify(this.response));
        var modelSet = this.response;
        var allInputs = modelSet.metadata.inputNames;
        modelSet.modelSet.forEach(function (value) {
            var numInputs = value.numInputs;
            var whichInputs = new Module.VectorInt();
            switch (value.modelType) {
                case 'kNN classification':
                    var neighbours = new Module.TrainingSet();
                    var numExamples = value.numExamples;
                    var k = value.k;
                    var numClasses = value.numClasses;

                    for (var i = 0; i < allInputs.length; ++i) {
                        if (value.inputNames.includes(allInputs[i])) {
                            whichInputs.push_back(i);
                        }
                    }

                    var myKnn = new Module.KnnClassification(numInputs, whichInputs, neighbours, k);
                    value.examples.forEach(function (value) {
                        var features = new Module.VectorDouble();
                        for (var i = 0; i < numInputs; ++i) {
                            features.push_back(parseFloat(value.features[i]));
                        }
                        myKnn.addNeighbour(parseInt(value.class), features);
                    });
                    that.addkNNModel(myKnn);
                    break;
                case 'Neural Network':
                    var numLayers = value.numHiddenLayers;
                    var numNodes = value.numHiddenNodes;
                    var weights = new Module.VectorDouble();
                    var wHiddenOutput = new Module.VectorDouble();
                    var inMax = new Module.VectorDouble();
                    var inMin = new Module.VectorDouble();
                    var outMax = value.outMax;
                    var outMin = value.outMin;

                    var localWhichInputs = [];
                    for (var i = 0; i < allInputs.length; ++i) {
                        //console.log('allInputs[', i, '] = ', allInputs[i]);
                        //console.log(value.inputNames);
                        if (value.inputNames.includes(allInputs[i])) {
                            whichInputs.push_back(i);
                            localWhichInputs.push(i);
                        }
                    }

                    var currentLayer = 0;
                    value.nodes.forEach(function (value, i) {
                        if (value.name === 'Linear Node 0') { //Output Node
                            for (var j = 1; j <= numNodes; ++j) {
                                var whichNode = 'Node ' + (j + (numNodes * (numLayers - 1)));
                                wHiddenOutput.push_back(parseFloat(value[whichNode]));
                                //console.log("pushing output ", value[whichNode]);
                            }
                            wHiddenOutput.push_back(parseFloat(value.Threshold));
                        } else {
                            currentLayer = Math.floor((i - 1) / numNodes);
                            if (currentLayer < 1) { //Nodes connected to input
                                for (var j = 0; j < numInputs; ++j) {
                                    //console.log('j ', j, 'whichInputs ', localWhichInputs[j]);
                                    //console.log("pushing", value['Attrib ' + allInputs[j]]);
                                    weights.push_back(parseFloat(value['Attrib ' + allInputs[localWhichInputs[j]]]));
                                }
                            } else { //Hidden Layers
                                for (var j = 1; j <= numNodes; ++j) {
                                    weights.push_back(parseFloat(value['Node ' + (j + (numNodes * (currentLayer - 1)))]));
                                }
                            }
                            weights.push_back(parseFloat(value.Threshold));
                        }
                    });

                    for (var j = 0; j < numInputs; ++j) {
                        inMin.push_back(value.inMins[j]);
                        inMax.push_back(value.inMaxes[j]);
                    }

                    var myNN = new Module.NeuralNetwork(numInputs, whichInputs, numLayers, numNodes, weights, wHiddenOutput, inMax, inMin, outMax, outMin);
                    that.addNNModel(myNN);
                    break;
                default:
                    console.warn('unknown model type ', value.modelType);
                    break;
            }
        });
    };
    request.send(null);
};

Module.ModelSet.prototype.addNNModel = function (model) {
    console.log('Adding NN model');
    this.myModelSet.push(model);
};

Module.ModelSet.prototype.addkNNModel = function (model) {
    console.log('Adding kNN model');
    this.myModelSet.push(model);
};

Module.ModelSet.prototype.process = function (input) {
    var modelSetInput = new Module.VectorDouble();
    for (var i = 0; i < input.length; ++i) {
        modelSetInput.push_back(input[i]);
    }
    var output = [];
    //noinspection JSDuplicatedDeclaration
    for (var i = 0; i < this.myModelSet.length; ++i) {
        output.push(this.myModelSet[i].processInput(modelSetInput));
    }
    return output;
};
