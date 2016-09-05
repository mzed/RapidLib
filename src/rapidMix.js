var rapidMix = {};

//rapidMix namespace

rapidMix.VectorInt = Module.VectorInt;
rapidMix.VectorDouble = Module.VectorDouble;
rapidMix.VectorNeighbour = Module.VectorNeighbour;

rapidMix.knnClassification = Module.knnClassification;
rapidMix.neuralNetwork = Module.neuralNetwork;

rapidMix.modelSet = function() {
    console.log("creating model set");
    this.myModelSet = [];
}

rapidMix.modelSet.prototype.loadJSON = function (url, contextIn) {
    var that = this;
    var request = new XMLHttpRequest();
    request.open("GET", url, true);
    request.responseType = "json";
    request.onload = function () {
        //console.log("loaded", JSON.stringify(this.response));
        var modelSet = this.response;
        modelSet.modelSet.forEach(function (value, i) {
            switch (value.modelType) {
                case 'kNN classification':
                    var numInputs = value.numInputs;
                    var whichInputs = new rapidMix.VectorInt(); //TODO: compare to all inputs
                    var neighbours = new rapidMix.VectorNeighbour();
                    var numExamples = value.numExamples
                    var k = value.k;
                    var numClasses = value.numClasses;

                    //test crap
                    whichInputs.push_back(0);
                    whichInputs.push_back(1);

                    var myKnn = new rapidMix.knnClassification(numInputs, whichInputs, neighbours, numExamples, k, numClasses);
                    value.examples.forEach(function (value, i) {
                        var features = new rapidMix.VectorDouble();
                        for (var i = 0; i < numInputs; ++i) {
                            features.push_back(parseFloat(value.features[i]));
                        }
                        myKnn.addNeighbour(parseInt(value.class), features);
                    });
                    that.addkNNModel(myKnn);
                    break;
                case 'Neural Network':
                    var numInputs = value.numInputs;
                    var whichInputs = new rapidMix.VectorInt(); //TODO: compare all inputs to this set
                    var weights = new rapidMix.VectorDouble();
                    var numNodes = value.numHiddenNodes;
                    var wHiddenOutput = new rapidMix.VectorDouble();
                    var inMax = new rapidMix.VectorDouble();
                    var inMin = new rapidMix.VectorDouble();
                    var outMax = value.outMax;
                    var outMin = value.outMin;

                    ///Test crap
                    whichInputs.push_back(0);
                    whichInputs.push_back(1);

                    value.nodes.forEach(function (value, i) {
                        if (value.name === 'Linear Node 0') {
                            wHiddenOutput.push_back(parseFloat(value['Node 1']));//FIXME
                            wHiddenOutput.push_back(parseFloat(value['Node 2']));
                            wHiddenOutput.push_back(parseFloat(value.Threshold));
                        } else {
                            weights.push_back(parseFloat(value['Attrib inputs-1']));//FIXME
                            weights.push_back(parseFloat(value['Attrib inputs-2']));
                            weights.push_back(parseFloat(value.Threshold));
                        }
                    });

                    for(var i = 0; i < numInputs; ++i) {
                        inMin.push_back(value.inMins[i]);
                        inMax.push_back(value.inMaxes[i]);
                    }

                    var myNN = new rapidMix.neuralNetwork(numInputs, whichInputs, numNodes, weights, wHiddenOutput, inMax, inMin, outMax, outMin);
                    that.addNNModel(myNN);
                    break;
                default:
                    console.log('unknown model type ', i);
                    break;
            };
        });
    };
    request.send(null);
}

rapidMix.modelSet.prototype.addNNModel = function (model) {
    console.log('Adding NN model');
    this.myModelSet.push(model);
}

rapidMix.modelSet.prototype.addkNNModel = function (model) {
    console.log('Adding kNN model');
    this.myModelSet.push(model);
}

rapidMix.modelSet.prototype.processInput = function (input) {
    var nnInput = new rapidMix.VectorDouble();
    nnInput.push_back(input[0]);//FIXME: Num inputs
    nnInput.push_back(input[1]);
    var output = [];
    output.push(this.myModelSet[0].processInput(nnInput));
        output.push(this.myModelSet[1].processInput(nnInput));
    return output;
}