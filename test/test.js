"use strict";
let expect = require('chai').expect;
var rapidMix = require('../rapidLib/RapidLib.js');

var jsons = require('./modelSetDescription.json');

var testSet = [
    {
        input: [0, 0],
        output: [0]
    },
    {
        input: [0, 1],
        output: [1]
    },
    {
        input: [1, 0],
        output: [1]
    },
    {
        input: [1, 1],
        output: [2]
    },
];

var testSet2 = [
    {
        input: [0, 0],
        output: [0, 9]
    },
    {
        input: [0, 1],
        output: [1, 2]
    },
    {
        input: [1, 0],
        output: [1, 2]
    },
    {
        input: [1, 1],
        output: [2, 4]
    },
];

var badSet = [
    {
        input: [1, 2, 3, 4],
        output: [5]
    },
    {
        input: [6],
        output: [7]
    }
];

var testSeries = [
    {
        input: [1., 5.],
        output: []
    },
    {
        input: [2., 4.],
        output: []
    },
    {
        input: [3., 3.],
        output: []
    },
    {
        input: [4., 2.],
        output: []
    },
    {
        input: [1., 5.],
        output: []
    }
];

var testSeries2 = [
    {
        input: [1., 4.]
    },
    {
        input: [2., -3.]
    },
    {
        input: [1., 5.]
    },
    {
        input: [-2., 1.]
    }
];

describe('RapidLib Machine Learning', function () {
    describe('Regression', function () {
        var myRegression = new rapidMix.Regression();
        it('should create a new Regression object', function () {
            expect(myRegression).to.be.an.instanceof(rapidMix.Regression);
        });
        it('should create a new object with a modelSet', function () {
            expect(myRegression).to.have.property('modelSet');
        });
        it('should reject malformed training data', function () {
            let trained = myRegression.train(badSet);
            expect(trained).to.be.false;
        });
        it('should train the modelSet with good data', function () {
            let trained = myRegression.train(testSet);
            expect(trained).to.be.true;
        });
        it('process() should return expected results', function () {
            let response1 = myRegression.process([0, 0]);
            expect(response1[0]).to.be.below(0.000001); //close enough
            let response2 = myRegression.process([0.2789, 0.4574]);
            expect(response2[0]).to.equal(0.6737399669524929);
            let response3 = myRegression.process(0.9, 0.7); //likes a list as well as an array
            expect(response3[0]).to.equal(1.6932444207337964);
        });
        it('should return zero on input that doesn\'t match numInputs', function () {
            let response1 = myRegression.process([33, 2, 44, 9]);
            expect(response1[0]).to.equal(0);
            let response2 = myRegression.process([1]);
            expect(response2[0]).to.equal(0);
        });
        it('should work with multiple outputs', function () {
            let myReg2 = new rapidMix.Regression();
            expect(myReg2).to.be.an.instanceof(rapidMix.Regression);
            expect(myReg2).to.have.property('modelSet');
            let trained = myReg2.train(testSet2);
            expect(trained).to.be.true;
            let response1 = myReg2.process([0, 0]);
            expect(response1[0]).to.be.below(0.000001); //close enough
            expect(response1[1]).to.be.above(8.99); //close enough
            let response2 = myReg2.process([0.2789, 0.4574]);
            expect(response2[0]).to.equal(0.6737399669524929);
            expect(response2[1]).to.equal(2.2184955630745575);
            let response3 = myReg2.process(0.9, 0.7); //NB: this is not an array
            expect(response3[0]).to.equal(1.6932444207337964);
            expect(response3[1]).to.equal(3.40508769608171);
        });
        it('can be initialized', function () {
            myRegression.reset();
            let response2 = myRegression.process([0.2789, 0.4574]);
            expect(response2[0]).to.equal(0); //initialized models return 0
        })
    });

    describe('KNN Classification', function () {
        let myClassification = new rapidMix.Classification();
        it('should create a new Classification object', function () {
            expect(myClassification).to.be.an.instanceof(rapidMix.Classification);
        });
        it('should create a new object with a modelSet', function () {
            expect(myClassification).to.have.property('modelSet');
        });
        it('should reject malformed training data', function () {
            let trained = myClassification.train(badSet);
            expect(trained).to.be.false;
        });
        it('should train the modelSet with good data', function () {
            let trained = myClassification.train(testSet);
            expect(trained).to.be.true;
        });
        it('process() should return expected results', function () {
            let response1 = myClassification.process([0, 0]);
            expect(response1[0]).to.be.equal(0);
            let response2 = myClassification.process([0.8789, 0.1574]);
            expect(response2[0]).to.equal(1);
            let response3 = myClassification.process([0.9, 0.7]);
            expect(response3[0]).to.equal(2);
        });
        it('should return zero on input that doesn\'t match numInputs', function () {
            let response1 = myClassification.process([33, 2, 44, 9]);
            expect(response1[0]).to.equal(0);
            let response2 = myClassification.process([1]);
            expect(response2[0]).to.equal(0);
        });
        it('should work with multiple outputs', function () {
            let myClass2 = new rapidMix.Classification();
            expect(myClass2).to.be.an.instanceof(rapidMix.Classification);
            expect(myClass2).to.have.property('modelSet');
            let trained = myClass2.train(testSet2);
            expect(trained).to.be.true;
            let response1 = myClass2.process([0, 0]);
            expect(response1[0]).to.equal(0);
            expect(response1[1]).to.equal(9);
            let response2 = myClass2.process([0.8789, 0.1574]);
            expect(response2[0]).to.equal(1);
            expect(response2[1]).to.equal(2);
            let response3 = myClass2.process([0.9, 0.7]);
            expect(response3[0]).to.equal(2);
            expect(response3[1]).to.equal(4);
        });
        it('can be initialized', function () {
            myClassification.reset();
            let response2 = myClassification.process([0.2789, 0.4574]);
            expect(response2[0]).to.equal(0); //initialized models return 0
        });
        it('can get k', function () {
            let myClass = new rapidMix.Classification();
            myClass.train(testSet2);
            let response3 = myClass.getK();
            expect(response3[0]).to.equal(1);
        });
        it('can set k', function () {
            let myClass = new rapidMix.Classification();
            myClass.train(testSet2);
            myClass.setK(0, 3);
            let response3 = myClass.getK();
            expect(response3[0]).to.equal(3);
            myClass.setK(0, 999);
            response3 = myClass.getK();
            expect(response3[0]).to.equal(4);
        });
    });
/*
    describe('SVM Classification', function () {
        let mySVM = new rapidMix.Classification(rapidMix.ClassificationTypes.SVM);
        it('should create a new Classification object', function () {
            expect(mySVM).to.be.an.instanceof(rapidMix.Classification);
        });
        it('should create a new object with a modelSet', function () {
            expect(mySVM).to.have.property('modelSet');
        });

        it('should reject malformed training data', function () {
            let trained = mySVM.train(badSet);
            expect(trained).to.be.false;
        });

        it('should train the modelSet with good data', function () {
            let trained = mySVM.train(testSet);
            expect(trained).to.be.true;
        });

        it('process() should return expected results' , function () {
            let response1 = mySVM.process([0, 0]);
            expect(response1[0]).to.be.equal(0);
            let response2 = mySVM.process([0.8789, 0.1574]);
            expect(response2[0]).to.equal(1);
            let response3 = mySVM.process([0.9, 0.7]);
            expect(response3[0]).to.equal(2);
        });
        it('should return zero on input that doesn\'t match numInputs', function () {
            let response1 = mySVM.process([33, 2, 44, 9]);
            expect(response1[0]).to.equal(0);
            let response2 = mySVM.process([1]);
            expect(response2[0]).to.equal(0);
        });

         it('should work with multiple outputs', function () {
         let mySVM2 = new rapidMix.Classification(rapidMix.ClassificationTypes.SVM);
         expect(mySVM2).to.be.an.instanceof(rapidMix.Classification);
         expect(mySVM2).to.have.property('modelSet');
         let trained = mySVM2.train(testSet2);
         expect(trained).to.be.true;
         let response1 = mySVM2.process([0, 0]);
         expect(response1[0]).to.equal(0);
         expect(response1[1]).to.equal(9);
         let response2 = mySVM2.process([0.8789, 0.1574]);
         expect(response2[0]).to.equal(1);
         expect(response2[1]).to.equal(2);
         let response3 = mySVM2.process([0.9, 0.7]);
         expect(response3[0]).to.equal(2);
         expect(response3[1]).to.equal(4);
         });
        it('can be initialized', function () {
            mySVM.reset();
            let response2 = mySVM.process([0.2789, 0.4574]);
            expect(response2[0]).to.equal(0); //initialized models return 0
        });
    });
*/

    describe('ModelSet', function () {
        let myModelSet = new rapidMix.ModelSet();
        it('should create a new ModelSet object', function () {
            expect(myModelSet).to.be.an.instanceof(rapidMix.ModelSet);
        });
        it('should create a new object with a modelSet', function () {
            expect(myModelSet).to.have.property('modelSet');
        });
        it('should load from JSON?', function () {
            //
            // TODO: need to stub XMLHttpRequest
            //myModelSet.loadJSON('modelSetDescription.json');
        });
        it('should process input');
    });

    describe('seriesClassification', function () {
        let myDTW = new rapidMix.SeriesClassification();
        it('should create a new seriesClassification object', function () {
            expect(myDTW).to.be.an.instanceof(rapidMix.SeriesClassification);
        });

        it('should allow me to add a series', function () {
            let added = myDTW.addSeries(testSeries);
            expect(added).to.be.true;
        });
        let myDTW2 = new rapidMix.SeriesClassification();
        let myDTW3 = new rapidMix.SeriesClassification();

        myDTW2.addSeries(testSeries);
        myDTW2.addSeries(testSeries2);

        let seriesSet = [testSeries, testSeries2];
        myDTW3.train(seriesSet);

        it('should let me train on a series set', function () {
            expect(myDTW3.run(testSeries2)).to.equal(1);
        });

        it('should correctly identify series 1', function () {
            expect(myDTW2.run(testSeries)).to.equal(0);
        });
       it('should correctly identify series 2', function () {
            expect(myDTW2.run(testSeries2)).to.equal(1);
        });
        it('should test clear');
    });
});

describe('RapidLib Signal Processing', function () {
    let myStream = new rapidMix.StreamProcess;
    let myStream10 = new rapidMix.StreamProcess(10);
    it('should create a new StreamProcess object', function () {
        expect(myStream).to.be.an.instanceof(rapidMix.StreamProcess);
        expect(myStream10).to.be.instanceof(rapidMix.StreamProcess);
    });

    myStream.clear();
    myStream.push(0);


    myStream.push(1);
    it('velocity should be 1', function () {
        expect(myStream.velocity()).to.equal(1);
    });

    let accelStream = new rapidMix.StreamProcess;
    accelStream.push(0);
    accelStream.push(0);
    accelStream.push(11);
    it('acceleration should be 10'/*, function () {
        expect(accelStream.acceleration()).to.equal(10);
    }*/);
    describe('when streaming to statStream', function () {
        let statStream = new rapidMix.StreamProcess(5);
        statStream.push(1.1);
        statStream.push(2.2);
        statStream.push(3.14);
        statStream.push(-4.3);
        statStream.push(33.9);
        it('should sum to 36.04', function () {
            expect(statStream.sum()).to.equal(36.04);
        });
        it('the mean should be 7.208', function () {
            expect(statStream.mean()).to.equal(7.208);
        });
        it('the standard deviation should be 13.592889906123716', function () {
            expect(statStream.standardDeviation()).to.equal(13.592889906123716);
        });
        it('the should rms be 15.385770048977072', function () {
            expect(statStream.rms()).to.equal(15.385770048977072);
        });
    });
    describe('when streaming to velStream', function () {
        let velStream = new rapidMix.StreamProcess(5);
        velStream.push(1.1);
        velStream.push(2.2);
        velStream.push(3.14);
        velStream.push(-4.3);
        velStream.push(33.9);
        it('check minVelocity', function () {
            expect(velStream.minVelocity()).to.equal(-7.4399999999999995);
        });
        it('check maxVelocity', function () {
            expect(velStream.maxVelocity()).to.equal(38.199999999999996);
        });
        it('check minAcceleration', function () {
            expect(velStream.minAcceleration()).to.equal(-8.379999999999999);
        });
        it('check maxAcceleration', function () {
            expect(velStream.maxAcceleration()).to.equal(45.63999999999999);
        });
    });
});