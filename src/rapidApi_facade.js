/*jshint esnext: true */
/*globals rapidMix*/

"use strict";

rapidMix.Regression = function () {
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
            var whichInputs = new rapidMix.VectorInt;
            for (let i = 0; i < this.numInputs; ++i) {
                whichInputs.push_back(i);
            }
            this.model.push(new rapidMix.NeuralNetwork(this.numInputs, whichInputs, 1, this.numInputs));
            this.created = true;
            break;
        default:
            console.error('rapidMix regression takes 2 arguments: # of inputs, # of outputs');
    }
};

rapidMix.Regression.prototype = {
    train(trainingSet) {
        var start = new Date().getTime();
        if (this.created) {
            for (let i = 0; i < this.model.length; ++i) {
                let rmTrainingSet = new rapidMix.TrainingSet;
                for (let trainingExample of trainingSet) {
                    let tempDouble = new rapidMix.VectorDouble;
                    for (let i = 0; i < this.numInputs; ++i) {
                        tempDouble.push_back(trainingExample.input[i]);
                    }
                    var tempObj = {'input': tempDouble, 'output': trainingExample.output[0]};
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
            for (let trainingExample of trainingSet) {
                if (trainingExample.input.length != this.numInputs) {
                    console.error("training set examples have different numbers of inputs");
                    return false;
                }
                if (trainingExample.output.length != this.numOutputs) {
                    console.error("training set examples have different numbers of outputs");
                    return false;
                }
            }
            for (let i = 0; i < this.numOutputs; ++i) {
                var whichInputs = new rapidMix.VectorInt;
                for (let i = 0; i < this.numInputs; ++i) {
                    whichInputs.push_back(i);
                }
                this.model.push(new rapidMix.NeuralNetwork(this.numInputs, whichInputs, 1, this.numInputs))
            }
            this.created = true;
            return this.train(trainingSet);
        }
    },
    process(input) {
        if (this.created) {
            var returnArray = [];
            for (let i = 0; i < this.model.length; ++i) {
                let inputVector = new rapidMix.VectorDouble;
                for (let j = 0; j < input.length; ++j) {
                    inputVector.push_back(input[j]);
                }
                returnArray.push(this.model[i].process(inputVector));
            }
            return returnArray;
        } else {
            console.error("No trained model here");
            return false;
        }
    }
};

rapidMix.Classification = function () {
    this.model = [];
    this.numInputs = 0;
    this.numOutputs = 0;
    this.created = false;
    switch (arguments.length) {
        case 0:
            break;
        case 2:
            var trainingSet = new rapidMix.TrainingSet;
            this.numInputs = arguments[0];
            var whichInputs = new rapidMix.VectorInt;
            for (let i = 0; i < this.numInputs; ++i) {
                whichInputs.push_back(i);
            }
            this.numOutputs = arguments[1];
            this.model = [];
            for (let i = 0; i < this.numOutputs; ++i) {
                this.model.push(new rapidMix.KnnClassification(this.numInputs, whichInputs, trainingSet, 1));
            }
            this.created = true;
            break;
        default:
            console.error('rapidMix classification takes 2 arguments: # of inputs, # of outputs');
    }
};

rapidMix.Classification.prototype = {
    train(trainingSet) {
        var start = new Date().getTime();
        if (this.created) {
            for (let out = 0; out < this.numOutputs; ++out) {
                for (let i = 0; i < trainingSet.length; ++i) {
                    let features = new rapidMix.VectorDouble;
                    for (let j = 0; j < this.numInputs; ++j) {
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
            var rmTrainingSet = new rapidMix.TrainingSet();
            for (let trainingExample of trainingSet) {
                if (trainingExample.input.length != this.numInputs) {
                    console.error("training set examples have different numbers of inputs");
                    return false;
                }
                if (trainingExample.output.length != this.numOutputs) {
                    console.error("training set examples have different numbers of outputs");
                    return false;
                }
                var tempDouble = new rapidMix.VectorDouble;
                for (let i = 0; i < this.numInputs; ++i) {
                    tempDouble.push_back(trainingExample.input[i]);
                }
                var tempObj = {'input': tempDouble, 'output': trainingExample.output[0]};
                rmTrainingSet.push_back(tempObj);
            }

            var whichInputs = new rapidMix.VectorInt;
            for (let i = 0; i < this.numInputs; ++i) {
                whichInputs.push_back(i);
            }
            for (let i = 0; i < this.numOutputs; ++i) {
                this.model.push(new rapidMix.KnnClassification(this.numInputs, whichInputs, rmTrainingSet, 1));
            }
            this.created = true;
        }
    },
    process(input) {
        if (this.created) {
            var toKnn = new rapidMix.VectorDouble();
            for (let i = 0; i < this.numInputs; ++i) {
                toKnn.push_back(input[i]);
            }
            var returnArray = [];
            for (let i = 0; i < this.model.length; ++i) {
                returnArray.push(this.model[i].process(toKnn));
            }
            return returnArray;
        } else {
            console.error("No trained model here");
            return false;
        }
    }
};
