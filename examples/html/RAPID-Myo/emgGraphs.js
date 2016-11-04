//This tells Myo.js to create the web sockets needed to communnicate with Myo Connect


Myo.on('connected', function(){
	console.log('connected');
	this.streamEMG(true);

	setInterval(function(){

		if(realTimeUpdate){
			updateGraph(rawData);
			updateCompositeGraph(rawData);
		}
		
	}, 5)
})

Myo.connect('com.myojs.emgGraphs');


var rawData = [0,0,0,0,0,0,0,0];

Myo.on('emg', function(data){
	rawData = data;
})


var rangeEMG = 150;
var resolutionEMG = 200;
var emgGraphs = [];
var emgCompositeGraphs = [];


var graphData = [
	// Array(200) - init array with 200; quickly populating an array with default values:
	Array.apply("Channel 1", Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply("Channel 2", Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply("Channel 3", Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply("Channel 4", Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply("Channel 5", Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply("Channel 6", Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply("Channel 7", Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply("Channel 8", Array(resolutionEMG)).map(Number.prototype.valueOf,0)
]

var compositeGraphData = [
	Array.apply(null, Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(resolutionEMG)).map(Number.prototype.valueOf,0),
	Array.apply(null, Array(resolutionEMG)).map(Number.prototype.valueOf,0)
]


$(document).ready(function(){

	emgGraphs = graphData.map(function(podData, podIndex){
		return $('#pod' + podIndex).plot(formatFlotData(podData), {
			colors: ['#8aceb5'],
			xaxis: {
				show: false,
				min : 0,
				max : resolutionEMG
			},
			yaxis : {
				min : -rangeEMG,
				max : rangeEMG,
			},
			grid : {
				borderColor : "#8aceb5",
				borderWidth : 1
			},
			selection: {
				mode: "x"
			}
		}).data("plot");
	});

	emgCompositeGraphs = $('#emgComposite').plot( formatCompositeFlotData(graphData), {
		colors: [ '#04fbec', '#ebf1be', '#c14b2a', '#8aceb5'],
		xaxis: {
			show: true,
			min : 0,
			max : resolutionEMG
		},
		yaxis : {
			min : -rangeEMG,
			max : rangeEMG,
		},
		grid : {
			borderColor : "#427F78",
			borderWidth : 1,
		},
		selection: {
			mode: "x"
		},
		interaction: {
			redrawOverlayInterval: 1
		}
	}).data("plot");

});

var formatFlotData = function(data){
	return [data.map(function(val, index){
			return [index, val]
		})]
}


var updateGraph = function(emgData){

	graphData.map(function(data, index){
		graphData[index] = graphData[index].slice(1);
		graphData[index].push(emgData[index]);


		emgGraphs[index].setData(formatFlotData(graphData[index]));
		emgGraphs[index].draw();
	})

}


var formatCompositeFlotData = function(data){
	
	return Object.keys(graphData).map(function(axis){
		return {
			label : 'Channel ' + axis,
			data : graphData[axis].map(function(val, index){
				return [index, val]
			})
		}
	});
}



var updateCompositeGraph = function(emgData){
	
	// calls a provided callback function once for each element in an array, in order, and constructs a new array from the results. 
	// compositeGraphData.map(function(data, index){
	// 	compositeGraphData[index] = compositeGraphData[index].slice(1);
	// 	compositeGraphData[index].push(emgData[index]);

	emgCompositeGraphs.setData(formatCompositeFlotData(graphData));
	emgCompositeGraphs.draw();
}

/*




*/