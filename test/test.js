"use strict";
let expect = require('chai').expect;

var rapidMix = require('../wekiLib/RapidMixLib.js');
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

describe('RapidAPI', function () {

    describe('Regression', function () {
        var myRegression = new rapidMix.Regression();
        it('should create a new Regression object', function () {
            expect(myRegression).to.be.an.instanceof(rapidMix.Regression);
        });
        it('should create a new object with a modelSet', function () {
            expect(myRegression).to.have.property('modelSet');
        });
        it('should train the modelSet', function () {
            let trained = myRegression.train(testSet);
            expect(trained).to.be.true;
        });
        it('process() should return expected results', function () {
            let response1 = myRegression.process([0, 0]);
            expect(response1[0]).to.be.below(0.000001); //close enough
            let response2 = myRegression.process([0.2789, 0.4574]);
            expect(response2[0]).to.equal(0.6907738688673892);
            let response3 = myRegression.process([0.9, 0.7]);
            expect(response3[0]).to.equal(1.6666849608321699);

        });
    });

    describe('Classification', function () {
        var myClassification = new rapidMix.Classification();
        it('should create a new Classification object', function () {
            expect(myClassification).to.be.an.instanceof(rapidMix.Classification);
        });
        it('should create a new object with a modelSet', function () {
            expect(myClassification).to.have.property('modelSet');
        });
        it('should train the modelSet', function () {
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
    });
});