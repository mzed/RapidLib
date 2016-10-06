/* globals Module */

"use strict";

/**
 * Utility function to convert js objects into something emscripten likes
 * @param {Object} trainingSet - JS Object representing a training set
 * @property {function} Module.TrainingSet - constructor for emscripten version of this struct
 * @property {function} Module.VectorDouble - constructor for the emscripten version of std::vector<double>
 * @returns {Module.TrainingSet}
 */
Module.prepTrainingSet = function (trainingSet) {
    var rmTrainingSet = new Module.TrainingSet();
    for (var i = 0; i < trainingSet.length; ++i) {
        var tempInput = new Module.VectorDouble();
        var tempOutput = new Module.VectorDouble();
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

////////////////////////////////////////////////

/**
 * Creates a set of regression objects using the constructor from emscripten
 * @constructor
 * @property {function} Module.RegressionCpp - constructor from emscripten
 */
Module.Regression = function () {
    this.modelSet = new Module.RegressionCpp(); //TODO implement optional arguments
};

Module.Regression.prototype = {
    /**
     * Trains the models using the input. Starts training from the current state of the model: randomized or trained.
     * @param {Object} trainingSet - An array of training examples
     * @returns {Boolean} true indicates successful training
     */
    train: function (trainingSet) {
        //change to vectorDoubles and send in
        return this.modelSet.train(Module.prepTrainingSet(trainingSet));
    },
    /**
     * Runs feed-forward regression on input
     * @param {Array} input - An array of features to be processed. Non-arrays are converted.
     * @returns {Array} output - One number for each model in the set
     */
    process: function (input) {
        //I'll assume that the args should have been an array
        if (arguments.length > 1) {
            input = Array.from(arguments);
        }
        //change input to vectors of doubles
        var inputVector = new Module.VectorDouble();
        for (var i = 0; i < input.length; ++i) {
            inputVector.push_back(input[i]);
        }
        //get the output
        var outputVector = new Module.VectorDouble();
        outputVector = this.modelSet.process(inputVector);
        //change back to javascript array
        var output = [];
        for (var i = 0; i < outputVector.size(); ++i) {
            output.push(outputVector.get(i));
        }
        return output;
    }
};

/////////////////////////////////////////////////

/**
 * Creates a set of classification objects using the constructor from emscripten
 * @constructor
 * @property {function} Module.ClassificationCpp - constructor from emscripten
 */

Module.Classification = function () {
    this.modelSet = new Module.ClassificationCpp(); //TODO implement optional arguments
};

Module.Classification.prototype = {
    /**
     * Trains the models using the input. Clears previous training set.
     * @param {Object} trainingSet - An array of training examples.
     * @returns {Boolean} true indicates successful training
     */
    train: function (trainingSet) {
        return this.modelSet.train(Module.prepTrainingSet(trainingSet));
    },
    /**
     * Does classifications on an input vector.
     * @param {Array} input - An array of features to be processed. Non-arrays are converted.
     * @returns {Array} output - One number for each model in the set
     */
    process: function (input) {
        //I'll assume that the args should have been an array
        if (arguments.length > 1) {
            input = Array.from(arguments);
        }
        //change input to vectors of doubles
        var inputVector = new Module.VectorDouble();
        for (var i = 0; i < input.length; ++i) {
            inputVector.push_back(input[i]);
        }
        //get the output
        var outputVector = new Module.VectorDouble();
        outputVector = this.modelSet.process(inputVector);
        //change back to javascript array
        var output = [];
        for (var i = 0; i < outputVector.size(); ++i) {
            output.push(outputVector.get(i));
        }
        return output;
    }
};

//////////////////////////////////////////////////

/**
 * Creates a set of machine learning objects using constructors from emscripten. Could be any mix of regression and classification.
 * @constructor
 */
Module.ModelSet = function () {
    this.myModelSet = [];
    this.modelSet = new Module.ModelSetCpp();
};

/**
 * Trains the models using the input. Clears previous training set.
 * @param {string} url - JSON loaded from a model set description document.
 * @returns {Boolean} true indicates successful training
 */
Module.ModelSet.prototype.loadJSON = function (url) {
    var that = this;
    console.log('url ', url);
    var b64 = Module.getBase64(url);
    console.log('b64 ', b64);
    var request = new XMLHttpRequest();
    request.open("GET", 'modelSetDescription.json', true);
    request.responseType = "json";
    request.onload = function () {
        console.log("req", this);
        console.log("loaded", this.response);
        var modelSet = this.responseText;
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
    return true; //TODO: make sure this is true;
};

/**
 * Add a NN model to a modelSet. //TODO: this doesn't need it's own function
 * @param model
 */
Module.ModelSet.prototype.addNNModel = function (model) {
    console.log('Adding NN model');
    this.myModelSet.push(model);
};

/**
 * Add a kNN model to a modelSet. //TODO: this doesn't need it's own function
 * @param model
 */
Module.ModelSet.prototype.addkNNModel = function (model) {
    console.log('Adding kNN model');
    this.myModelSet.push(model);
};

/**
 * Applies regression and classification algorithms to an input vector.
 * @param {Array} input - An array of features to be processed.
 * @returns {Array} output - One number for each model in the set
 */
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

Module.getBase64 = function(str) {
    //check if the string is a data URI
    if (str.indexOf(';base64,') != -1 ) {
        //see where the actual data begins
        var dataStart = str.indexOf(';base64,') + 8;
        //check if the data is base64-encoded, if yes, return it
        // taken from
        // http://stackoverflow.com/a/8571649
        return str.slice(dataStart).match(/^([A-Za-z0-9+\/]{4})*([A-Za-z0-9+\/]{4}|[A-Za-z0-9+\/]{3}=|[A-Za-z0-9+\/]{2}==)$/) ? str.slice(dataStart) : false;
    }
    else return false;
}