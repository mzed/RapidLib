//This tells Myo.js to create the web sockets needed to communnicate with Myo Connect
// Myo.connect('com.myojs.emgGraphs');

var rangeG = 500;
var resolutionG = 100;

var rangeA = 3;
var resolutionA = 100;

var rangeO = 1;
var resolutionO = 100;

var range = 500;
var resolution = 100;

var graphGyro;
var graphAccel;
var graphOrientation;


var graphOrientationData;

Myo.on('gyroscope', function(quant){
	updateGyroGraph(quant);
	// console.log('gyro:');
	// console.log(quant);
})

Myo.on('accelerometer', function(quant){
	updateAccelGraph(quant);
	// console.log('accel:');
	// console.log(quant);
})

Myo.on('orientation', function(quant){

	updateOriGraph(quant);
	graphOrientationData = quant;
})



var arrayOfZeros = Array.apply(null, Array(resolution)).map(Number.prototype.valueOf,0);

var graphDataGyro = {
	x : arrayOfZeros.slice(0),
	y : arrayOfZeros.slice(0),
	z : arrayOfZeros.slice(0),
}

var graphDataAccel = {
	x : arrayOfZeros.slice(0),
	y : arrayOfZeros.slice(0),
	z : arrayOfZeros.slice(0),
}

var graphDataOrientation = {
	x : arrayOfZeros.slice(0),
	y : arrayOfZeros.slice(0),
	z : arrayOfZeros.slice(0),
	w : Array.apply(null, Array(resolution)).map(Number.prototype.valueOf,0)
}


$(document).ready(function(){

	graphOrientation = $('#orientationGraph').plot(formatOriFlotData(), {
		colors: [ '#04fbec', '#ebf1be', '#c14b2a', '#8aceb5'],
		xaxis: {
			show: false,
			min : 0,
			max : resolutionO
		},
		yaxis : {
			min : -rangeO,
			max : rangeO,
		},
		grid : {
			borderColor : "#427F78",
			borderWidth : 1
		},
		selection: {
			mode: "x"
		}
	}).data("plot");

	graphGyro = $('#gyroGraph').plot(formatGyroFlotData(), {
		colors: [ '#04fbec', '#ebf1be', '#c14b2a', '#8aceb5'],
		xaxis: {
			show: false,
			min : 0,
			max : resolutionG
		},
		yaxis : {
			min : -rangeG,
			max : rangeG,
		},
		grid : {
			borderColor : "#427F78",
			borderWidth : 1
		},
		selection: {
			mode: "x"
		}
	}).data("plot");

	graphAccel = $('#accelGraph').plot(formatAccelFlotData(), {
		colors: [ '#04fbec', '#ebf1be', '#c14b2a', '#8aceb5'],
		xaxis: {
			show: false,
			min : 0,
			max : resolutionA
		},
		yaxis : {
			min : -rangeA,
			max : rangeA,
		},
		grid : {
			borderColor : "#427F78",
			borderWidth : 1
		},
		selection: {
			mode: "x"
		}

	}).data("plot");


});


var formatGyroFlotData = function(){
	return Object.keys(graphDataGyro).map(function(axis){
		return {
			label : axis + ' axis',
			data : graphDataGyro[axis].map(function(val, index){
				return [index, val]
			})
		}
	});
}

var updateGyroGraph = function(gyroData){
	Object.keys(gyroData).map(function(axis){
		graphDataGyro[axis] = graphDataGyro[axis].slice(1);
		graphDataGyro[axis].push(gyroData[axis]);
	});

	if(realTimeUpdate){
		graphGyro.setData(formatGyroFlotData());
		graphGyro.draw();
	}
}


var formatAccelFlotData = function(){
	return Object.keys(graphDataAccel).map(function(axis){
		return {
			label : axis + ' axis',
			data : graphDataAccel[axis].map(function(val, index){
				return [index, val]
			})
		}
	});
}

var updateAccelGraph = function(accelData){
	Object.keys(accelData).map(function(axis){
		graphDataAccel[axis] = graphDataAccel[axis].slice(1);
		graphDataAccel[axis].push(accelData[axis]);
	});

	if(realTimeUpdate){
		graphAccel.setData(formatAccelFlotData());
		graphAccel.draw();
	}
}

var formatOriFlotData = function(){
	return Object.keys(graphDataOrientation).map(function(axis){
		return {
			label : axis + ' axis',
			data : graphDataOrientation[axis].map(function(val, index){
				return [index, val]
			})
		}
	});
}

var updateOriGraph = function(orientationData){
	Object.keys(orientationData).map(function(axis){
		graphDataOrientation[axis] = graphDataOrientation[axis].slice(1);
		graphDataOrientation[axis].push(orientationData[axis]);
	});

	if(realTimeUpdate){
		graphOrientation.setData(formatOriFlotData());
		graphOrientation.draw();
	}
}
