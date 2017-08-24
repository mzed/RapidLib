/* globals Module */

"use strict";

console.log("RapidLib 14.8.2017 13:23")

/**
 * Utility function to convert js objects into C++ trainingSets
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

Module.prepTrainingSeriesSet = function (trainingSeriesSet) {
    var rmTrainingSeriesSet = new Module.TrainingSeriesSet();
    for (var i = 0; i < trainingSeriesSet.length; ++i) {
        var input = new Module.VectorVectorDouble();
        for (var j = 0; j < trainingSeriesSet[i].input.length; ++j) {
            var tempVector = new Module.VectorDouble();
            for (var k = 0; k < trainingSeriesSet[i].input[j].length; ++k) {
                tempVector.push_back(parseFloat(trainingSeriesSet[i].input[j][k]));
            }
            input.push_back(tempVector);
        }
        var tempObj = {'input': input, 'label': trainingSeriesSet[i].label};
        rmTrainingSeriesSet.push_back(tempObj);
    }
    return rmTrainingSeriesSet;
};

/**
 * Utility function to add an empty output to a "training set" if it is undefined
 * @param jsInput
 * @returns {*}
 */

Module.checkOutput = function (jsInput) {
    for (var i = 0; i < jsInput.length; ++i) {
        if (typeof jsInput[i].output === "undefined") {
            jsInput[i].output = [];
        }
    }
    return jsInput;
};
////////////////////////////////////////////////   Regression

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
     * Trains the models using the input. Starts training from a randomized state.
     * @param {Object} trainingSet - An array of training examples
     * @returns {Boolean} true indicates successful training
     */
    train: function (trainingSet) {
        this.modelSet.reset();
        //change to vectorDoubles and send in
        return this.modelSet.train(Module.prepTrainingSet(trainingSet));
    },
    /**
     * Returns the model set to its initial configuration.
     * @returns {Boolean} true indicates successful initialization
     */
    reset: function () {
        return this.modelSet.reset();
    },
    /**
     * Runs feed-forward regression on input
     * @param {Array} input - An array of features to be processed. Non-arrays are converted.
     * @returns {Array} output - One number for each model in the set
     */
    run: function (input) {
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
        outputVector = this.modelSet.run(inputVector);
        //change back to javascript array
        var output = [];
        for (var i = 0; i < outputVector.size(); ++i) {
            output.push(outputVector.get(i));
        }
        return output;
    },
    /**
     * Deprecated! Use run() instead
     * @param input
     * @returns {Array}
     */
    process: function (input) {
        //return this.run(input); //Why doesn't this work? MZ
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
        outputVector = this.modelSet.run(inputVector);
        //change back to javascript array
        var output = [];
        for (var i = 0; i < outputVector.size(); ++i) {
            output.push(outputVector.get(i));
        }
        return output;
    }
};


/////////////////////////////////////////////////  Classification

/**
 * Creates a set of classification objects using the constructor from emscripten
 * @constructor
 * @property {function} Module.ClassificationCpp - constructor from emscripten
 * @param {string} [type] - which classification algorithm to use
 */

Module.Classification = function (type) {
    if (type) {
        this.modelSet = new Module.ClassificationCpp(type);
    } else {
        this.modelSet = new Module.ClassificationCpp();
    }
};

Module.Classification.prototype = {
    /**
     * Trains the models using the input. Clears previous training set.
     * @param {Object} trainingSet - An array of training examples.
     * @returns {Boolean} true indicates successful training
     */
    train: function (trainingSet) {
        this.modelSet.reset();
        return this.modelSet.train(Module.prepTrainingSet(trainingSet));
    },
    /**
     * Returns a vector of current k values for each model.
     * @returns {Array} k values
     */
    getK: function () {
        var outputVector = new Module.VectorInt();
        outputVector = this.modelSet.getK();
        var output = [];
        for (var i = 0; i < outputVector.size(); ++i) {
            output.push(outputVector.get(i));
        }
        return output;
    },
    /**
     * Sets the k values for a particular model model.
     * @param {Number} whichModel - which model
     * @param {Number} newK - set K to this value
     */
    setK: function (whichModel, newK) {
        this.modelSet.setK(whichModel, newK);
    },
    /**
     * Returns the model set to its initial configuration.
     * @returns {Boolean} true indicates successful initialization
     */
    reset: function () {
        return this.modelSet.reset();
    },
    /**
     * Does classifications on an input vector.
     * @param {Array} input - An array of features to be processed. Non-arrays are converted.
     * @returns {Array} output - One number for each model in the set
     */
    run: function (input) {
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
        outputVector = this.modelSet.run(inputVector);
        //change back to javascript array
        var output = [];
        for (var i = 0; i < outputVector.size(); ++i) {
            output.push(outputVector.get(i));
        }
        return output;
    },
    /**
     * Deprecated! USe run() instead
     * @param input
     */
    process: function (input) {
        //return this.run(input); //why doesn't this work?
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
        outputVector = this.modelSet.run(inputVector);
        //change back to javascript array
        var output = [];
        for (var i = 0; i < outputVector.size(); ++i) {
            output.push(outputVector.get(i));
        }
        return output;
    }
};

//////////////////////////////////////////////////  ModelSet

/**
 * Creates a set of machine learning objects using constructors from emscripten. Could be any mix of regression and classification.
 * This is only useful when importing JSON from Wekinator.
 * @constructor
 */
Module.ModelSet = function () {
    this.myModelSet = [];
    this.modelSet = new Module.ModelSetCpp();
};

/**
 * Creates a model set populated with models described in a JSON document.
 * This only works in documents that are part of a CodeCircle document.
 * @param {string} url - JSON loaded from a model set description document.
 * @returns {Boolean} true indicates successful training
 */
Module.ModelSet.prototype = {
    loadJSON: function (url) {
        var that = this;
        console.log('url ', url);
        var request = new XMLHttpRequest();
        request.open("GET", url, true);
        request.responseType = "json";
        request.onload = function () {
            var modelSet = this.response;
            console.log("loaded: ", modelSet);
            var allInputs = modelSet.metadata.inputNames;
            modelSet.modelSet.forEach(function (value) {
                var numInputs = value.numInputs;
                var whichInputs = new Module.VectorInt();
                switch (value.modelType) {
                    case 'kNN classification':
                        var neighbours = new Module.TrainingSet();
                        var k = value.k;
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
                        var inRanges = new Module.VectorDouble();
                        var inBases = new Module.VectorDouble();

                        var localWhichInputs = [];
                        for (var i = 0; i < allInputs.length; ++i) {
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
                                }
                                wHiddenOutput.push_back(parseFloat(value.Threshold));
                            } else {
                                currentLayer = Math.floor((i - 1) / numNodes); //FIXME: This will break if node is out or order.
                                if (currentLayer < 1) { //Nodes connected to input
                                    for (var j = 0; j < numInputs; ++j) {
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

                        for (var i = 0; i < numInputs; ++i) {
                            inRanges.push_back(value.inRanges[i]);
                            inBases.push_back(value.Bases[i]);
                        }

                        var outRange = value.outRange;
                        var outBase = value.outBase;

                        var myNN = new Module.NeuralNetwork(numInputs, whichInputs, numLayers, numNodes, weights, wHiddenOutput, inRanges, inBases, outRange, outBase);
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
    },
    /**
     * Add a NN model to a modelSet. //TODO: this doesn't need it's own function
     * @param model
     */
    addNNModel: function (model) {
        console.log('Adding NN model');
        this.myModelSet.push(model);
    },
    /**
     * Add a kNN model to a modelSet. //TODO: this doesn't need it's own function
     * @param model
     */
    addkNNModel: function (model) {
        console.log('Adding kNN model');
        this.myModelSet.push(model);
    },
    /**
     * Applies regression and classification algorithms to an input vector.
     * @param {Array} input - An array of features to be processed.
     * @returns {Array} output - One number for each model in the set
     */
    run: function (input) {
        var modelSetInput = new Module.VectorDouble();
        for (var i = 0; i < input.length; ++i) {
            modelSetInput.push_back(input[i]);
        }
        var output = [];
        for (var i = 0; i < this.myModelSet.length; ++i) {
            output.push(this.myModelSet[i].run(modelSetInput));
        }
        return output;
    },
    /**
     * Deprecated! Use run() instead.
     * @param {Array} input - An array of features to be processed
     * @returns {Array} output - One number for each model in the set
     */
    process: function (input) {
        return this.run(input);
    }
};


////////////////////////////////////////////////

/**
 * Creates a series classification object using the constructor from emscripten
 * @constructor
 * @property {function} Module.SeriesClassificationCpp - constructor from emscripten
 */
Module.SeriesClassification = function () {
    this.seriesClassification = new Module.SeriesClassificationCpp(); //TODO implement optional arguments
};

Module.SeriesClassification.prototype = {
    /**
     * Adds a series to the array examples
     * @param {Object} newSeries - An array of arrays
     * @returns {Number} - index of the example series that best matches the input
     */
    // addSeries: function (newSeries) {
    //     newSeries = Module.checkOutput(newSeries);
    //     return this.seriesClassification.addTrainingSet(Module.prepTrainingSet(newSeries));
    // },
    /**
     * Resets the model, and adds a set of series to be evaluated
     * @param {Object} newSeriesSet - an array of objects, each with input: <array of arrays> and label: <string>
     * @return {Boolean} True indicates successful training.
     */
    train: function (newSeriesSet) {
        this.reset();
        this.seriesClassification.trainLabel(Module.prepTrainingSeriesSet(newSeriesSet));
        //     for (var i = 0; i < newSeriesSet.length; ++i) {
        //         newSeriesSet[i] = Module.checkOutput(newSeriesSet[i]);
        //         this.seriesClassification.addTrainingSet(Module.prepTrainingSet(newSeriesSet[i]));
        //     }
        return true;
    },
    /**
     * Returns the model set to its initial configuration.
     * @returns {Boolean} true indicates successful initialization
     */
    reset: function () {
        return this.seriesClassification.reset();
    },
    /**
     * Evaluates an input series and returns the index of the closet example
     * @param {Object} inputSeries - an array of arrays
     * @returns {Number} The index of the closest matching series
     */
    run: function (inputSeries) {
        var vecInputSeries = new Module.VectorVectorDouble();
        for (var i = 0; i < inputSeries.length; ++i) {
            var tempVector = new Module.VectorDouble();
            for (var j = 0; j < inputSeries[i].length; ++j) {
                tempVector.push_back(inputSeries[i][j]);
            }
            vecInputSeries.push_back(tempVector);
        }
        return this.seriesClassification.runLabel(vecInputSeries);

    },
    /**
     * Deprecated! Use run()
     * @param inputSeries
     * @returns {Number}
     */
    process: function (inputSeries) {
        return this.run(inputSeries);
    },
    /**
     * Returns an array of costs to match the input series to each example series. A lower cost is a closer match
     * @param {Array} [inputSeries] - An array of arrays to be evaluated. (Optional)
     * @returns {Array}
     */
    getCosts: function (inputSeries) {
        if (inputSeries) {
            inputSeries = Module.checkOutput(inputSeries);
            this.seriesClassification.runTrainingSet(Module.prepTrainingSet(inputSeries));
        }
        var returnArray = [];
        var VecDouble = this.seriesClassification.getCosts();
        for (var i = 0; i < VecDouble.size(); ++i) {
            returnArray[i] = VecDouble.get(i);
        }
        return returnArray;
    }
};

/////////////////////////////////////////////////

/**
 * Creates a circular buffer that can return various statistics
 * @constructor
 * @param {number} [windowSize=3] - specify the size of the buffer
 * @property {function} Module.rapidStreamCpp - constructor from emscripten
 */

Module.StreamBuffer = function (windowSize) {
    if (windowSize) {
        this.rapidStream = new Module.RapidStreamCpp(windowSize);
    } else {
        this.rapidStream = new Module.RapidStreamCpp();
    }
};

Module.StreamBuffer.prototype = {
    /**
     * Add a value to a circular buffer whose size is defined at creation.
     * @param {number} input - value to be pushed into circular buffer.
     */
    push: function (input) {
        this.rapidStream.pushToWindow(parseFloat(input));
    },
    /**
     * Resets all the values in the buffer to zero.
     */
    reset: function () {
        this.rapidStream.clear();
    },
    /**
     * Calculate the first-order difference (aka velocity) between the last two inputs.
     * @return {number} difference between last two inputs.
     */
    velocity: function () {
        return this.rapidStream.velocity();
    },
    /**
     * Calculate the second-order difference (aka acceleration) over the last three inputs.
     * @return {number} acceleration over the last three inputs.
     */
    acceleration: function () {
        return this.rapidStream.acceleration();
    },
    /**
     * Find the minimum value in the buffer.
     * @return {number} minimum.
     */
    minimum: function () {
        return this.rapidStream.minimum();
    },
    /**
     * Find the maximum value in the buffer.
     * @return {number} maximum.
     */
    maximum: function () {
        return this.rapidStream.maximum();
    },
    /**
     * Calculate the sum of all values in the buffer.
     * @return {number} sum.
     */
    sum: function () {
        return this.rapidStream.sum();
    },
    /**
     * Calculate the mean of all values in the buffer.
     * @return {number} mean.
     */
    mean: function () {
        return this.rapidStream.mean();
    },
    /**
     * Calculate the standard deviation of all values in the buffer.
     * @return {number} standard deviation.
     */
    standardDeviation: function () {
        return this.rapidStream.standardDeviation();
    },
    /**
     * Calculate the root mean square of the values in the buffer
     * @return {number} rms
     */
    rms: function () {
        return this.rapidStream.rms();
    },
    /**
     * Calculate the minimum first-order difference over consecutive inputs in the buffer.
     * @return {number} minimum velocity.
     */
    minVelocity: function () {
        return this.rapidStream.minVelocity();
    },
    /**
     * Calculate the maximum first-order difference over consecutive inputs in the buffer.
     * @return {number} maximum velocity.
     */
    maxVelocity: function () {
        return this.rapidStream.maxVelocity();
    },
    /**
     * Calculate the minimum second-order difference over consecutive inputs in the buffer.
     * @return {number} minimum acceleration.
     */
    minAcceleration: function () {
        return this.rapidStream.minAcceleration();
    },
    /**
     * Calculate the maximum second-order difference over consecutive inputs in the buffer.
     * @return {number} maximum acceleration.
     */
    maxAcceleration: function () {
        return this.rapidStream.maxAcceleration();
    }
};