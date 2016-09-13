/*jshint esnext: true */
/*globals rapidMix, synaptic */

/////////////////////////////////////////Façade for synaptic

rapidMix.Regression = function () {
    this.model;
    this.trainer;
    this.numInputs;
    this.numOutputs;
    this.created = false;
    switch (arguments.length) {
        case 0:
            break;
        case 2:
            if (arguments[1] === 1) {
                //One output. Create a single model and attach a trainer to it
                this.model = new synaptic.Architect.Perceptron(arguments[0], 3, 1); //TODO make this more like Wekinator
                this.trainer = new synaptic.Trainer(this.model);
                this.created = true;
                this.numOutputs = 1;
            } else {
                this.created = false;
                this.numOutputs = arguments[1];
                //create a model set with one model per output
            }
            break;
        default:
            console.error('rapidMix regression takes 2 arguments: # of inputs, # of outputs');
    }
};

rapidMix.Regression.prototype = {
    train(trainingSet) {
        if (this.created) {
            return this.trainer.train(trainingSet);
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
            if (this.numOutputs === 1) {
                //One output. Create a single model and attach a trainer to it
                this.model = new synaptic.Architect.Perceptron(this.numInputs, 3, 1); //TODO make this more like Wekinator
                this.trainer = new synaptic.Trainer(this.model);
                this.created = true;
                return this.trainer.train(trainingSet);
            } else {
                //create model set and train it
            }
        }
    },
    process(input) {
        if (this.created) {
            if (this.numOutputs === 1) {
                return this.model.activate(input);
            } else {
                console.log("Haven't implemented modelSets");
                return false;
            }
        } else {
            console.error("No trained model here");
            return false;
        }
    }
};

rapidMix.Classification = function () {
    this.numInputs;
    this.numOutputs;
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
            if (arguments[1] === 1) {
                this.model = new rapidMix.knnClassification(this.numInputs, whichInputs, neighbours, 0, 1);
                this.created = true;
            }
            break;
        default:
            console.error('rapidMix classification takes 2 arguments: # of inputs, # of outputs');
    }
};

rapidMix.Classification.prototype = {
    train(trainingSet) {
        var start = new Date().getTime();
        if (this.created) {
            for (let i = 0; i < trainingSet.length; ++i) {
                let features = new rapidMix.VectorDouble();
                for (let j = 0; j < this.numInputs; ++j) {
                    features.push_back(trainingSet[i].input[j]);
                }
                this.model.addNeighbour(trainingSet[i].output[0], features);
            }
            var end = new Date().getTime();
            var time = end - start;
            var result = {'examples': trainingSet.length, 'time': time};
            return result;
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
            if (this.numOutputs === 1) {
                //One output. Create a single model and train it
                var neighbours = new rapidMix.VectorNeighbour();
                var whichInputs = new rapidMix.VectorInt();
                for (let i = 0; i < this.numInputs; ++i) {
                    whichInputs.push_back(i);
                }
                this.model = new rapidMix.knnClassification(this.numInputs, whichInputs, neighbours, 0, 1);
                this.created = true;
                for (let i = 0; i < trainingSet.length; ++i) {
                    let features = new rapidMix.VectorDouble();
                    for (let j = 0; j < this.numInputs; ++j) {
                        features.push_back(trainingSet[i].input[j]);
                    }
                    this.model.addNeighbour(trainingSet[i].output[0], features);
                }
                var end = new Date().getTime();
                var time = end - start;
                var result = {'examples': trainingSet.length, 'time': time};
                return result;
            }
        }
    },
    process(input) {
        if (this.created) {
            var toKnn = new rapidMix.VectorDouble();
            for (let i = 0; i < this.numInputs; ++i) {
                toKnn.push_back(input[i]);
            }
            return this.model.processInput(toKnn);
        } else {
            console.error("No trained model here");
            return false;
        }
    }
};