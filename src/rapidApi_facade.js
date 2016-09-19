/*jshint esnext: true */
/*globals rapidMix, synaptic */

"use strict";

/////////////////////////////////////////Façade for synaptic

rapidMix.Regression = function () {
    this.model = [];
    this.trainer = [];
    this.numInputs = 0;
    this.numOutputs = 0;
    this.created = false;
    switch (arguments.length) {
        case 0:
            break;
        case 2:
            this.numInputs = arguments[0];
            this.numOutputs = arguments[1];
            for (let i = 0; i < this.numOutputs; ++i) {
                //this.model.push(new synaptic.Architect.Perceptron(this.numInputs, this.numInputs, 1));
                this.model.push(makeNN(this.numInputs));
                this.trainer.push(new synaptic.Trainer(this.model[i]));
            }
            this.created = true;
            break;
        default:
            console.error('rapidMix regression takes 2 arguments: # of inputs, # of outputs');
    }
};

rapidMix.Regression.prototype = {
    train(trainingSet) {
        if (this.created) {
            var returnObj;
            for (let i = 0; i < this.trainer.length; ++i) {
                let tempTrainingSet = [];
                for (let j = 0; j < trainingSet.length; ++j) {
                    let tempOutput = trainingSet[j].output[i];
                    tempTrainingSet[j] = {
                        input: trainingSet[j].input,
                        output: [tempOutput]
                    };
                }
                returnObj = this.trainer[i].train(tempTrainingSet);
                console.log(returnObj);
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
                //this.model.push(new synaptic.Architect.Perceptron(this.numInputs, this.numInputs, 1));
                this.model.push(makeNN(this.numInputs));
                this.trainer.push(new synaptic.Trainer(this.model[i]));
            }
            this.created = true;
            return this.train(trainingSet);
        }
    },
    process(input) {
        if (this.created) {
            var returnArray = [];
            for (let i = 0; i < this.model.length; ++i) {
                returnArray.push(this.model[i].activate(input));
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
            var neighbours = new rapidMix.VectorNeighbour();
            this.numInputs = arguments[0];
            var whichInputs = new rapidMix.VectorInt();
            for (let i = 0; i < this.numInputs; ++i) {
                whichInputs.push_back(i);
            }
            this.numOutputs = arguments[1];
            this.model = [];
            for (let i = 0; i < this.numOutputs; ++i) {
                this.model.push(new rapidMix.knnClassification(this.numInputs, whichInputs, neighbours, 1));
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
                    let features = new rapidMix.VectorDouble();
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

            var neighbours = new rapidMix.VectorNeighbour();
            var whichInputs = new rapidMix.VectorInt();
            for (let i = 0; i < this.numInputs; ++i) {
                whichInputs.push_back(i);
            }
            for (let i = 0; i < this.numOutputs; ++i) {
                this.model.push(new rapidMix.knnClassification(this.numInputs, whichInputs, neighbours, 1));
            }
            this.created = true;
            return this.train(trainingSet);
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
                returnArray.push(this.model[i].processInput(toKnn));
            }
            return returnArray;
        } else {
            console.error("No trained model here");
            return false;
        }
    }
};

function makeNN(inputs) {
    var outputLayer = new synaptic.Layer(1);
    outputLayer.set({
        squash: synaptic.Neuron.squash.IDENTITY,
        bias: 1
    });
    var hiddenLayer = new synaptic.Layer(inputs);
    var inputLayer = new synaptic.Layer(inputs);

    inputLayer.project(hiddenLayer);
    hiddenLayer.project(outputLayer);

    return new synaptic.Network({
        input: inputLayer,
        hidden: [hiddenLayer],
        output: outputLayer
    })
}