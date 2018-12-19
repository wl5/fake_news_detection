$(document).ready(function(){    
    namespace = '/demo_streaming'; // change to an empty string to use the global namespace
    
    var socket = io.connect('http://' + document.domain + ':' + location.port + namespace);

    var track_item = ["agree", "disagree", "discuss", "unrelated"]
    var count = {}
    track_item.forEach(function(term){
	count[term] = 0;
    });

    var w = 1100;
    var h = 900;
    var barPadding = 5;

    var svg = d3.select("body")
	.append("svg")
	.attr("width", w)
	.attr("height", h);

    var dataset = Object.keys(count).map(function(key){
	return count[key];
    });

    
    svg.selectAll("rect")
	.data(dataset)
	.enter()
	.append("rect")
	.attr("x", function(d, i){
	    return i * (w / dataset.length);
	})
	.attr("y", function(d){
	    return h - (d * 4);
	})
	.attr("width", w / dataset.length - barPadding)
	.attr("height", function(d){
	    return d * 4;
	})
	.attr("fill", function(d){
	    return "rgb(0, 0, " + (d * 10) + ")";
	});

    svg.selectAll("text")
	.data(dataset)
	.enter()
	.append("text")
	.text(function(d){
	    return d;
	})
	.attr("text-anchor", "middle")
	.attr("x", function(d, i){
	    return i * (w / dataset.length) + (w / dataset.length - barPadding) / 2;
	})
	.attr("y", function(d){
	    return h;
	})
	.attr("font-family", "sans-serif")
	.attr("font-size", "11px")
	.attr("fill", "white");
    
    socket.on('stream_channel', function(data) {
	count[data.pred] += 1;

	var dataset = Object.keys(count).map(function(key){
	    return count[key];
	});

	// update rectangles
	// count & update bar chart
	svg.selectAll("rect")
	    .data(dataset)
	    .transition()
	    .attr("x", function(d, i){
		return i * (w / dataset.length);
	    })
	    .attr("y", function(d){
		return h - (d * 4);
	    })
	    .attr("width", w / dataset.length - barPadding)
	    .attr("height", function(d){
		return d * 4;
	    })
	    .attr("fill", function(d){
		return "rgb(0, 0, " + (d * 10) + ")";
	    });
	svg.selectAll("text")
	    .data(dataset)
	    .transition()
	    .text(function(d, i){
		return track_item[i] + ": " + d;
	    })
	    .attr("text-anchor", "middle")
	    .attr("x", function(d, i){
		return i * (w / dataset.length) +
		    (w / dataset.length - barPadding) / 2;
	    })
	    .attr("y", function(d){
		return h - (d * 4);
	    })
	    .attr("font-family", "sans-serif")
	    .attr("font-size", "11px")
	    .attr("fill", "black");
    });
        //$('#log').text(msg.agree + ' ' + msg.disagree + ' ' + msg.discuss + ' ' + msg.unrelated);
});
