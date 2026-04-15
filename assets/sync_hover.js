window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        sync_plots: function(tsHover, mapId) {

            // No hover yet or map not ready
            if (!tsHover || !tsHover.points || !tsHover.points.length) {
                return window.dash_clientside.no_update;
            }
            
            const t = tsHover.points[0].customdata && tsHover.points[0].customdata[0];
            const e = tsHover.points[0].customdata && tsHover.points[0].customdata[1]; // the expocode
            if (!t) return window.dash_clientside.no_update;
    
            const gd = document.getElementById(mapId);
            if (!gd || !gd.children[1].data) return window.dash_clientside.no_update;
    
            const mapPlot = gd.children[1];
    
            // Find ALL map points whose customdata[0] matches the hovered time
            // (there may be multiple trajectories at the same timestamp)
            const curveNumber = 0; // px.scatter_geo produces a single trace by default
            
            const trace = mapPlot.data[curveNumber];
            if (!trace || !trace.customdata) return window.dash_clientside.no_update;
    
            const pointNumbers = [];
            for (let i = 0; i < trace.customdata.length; i++) {
                const cd = trace.customdata[i];
                if ((cd && cd[0] === t) && (cd && cd[1] === e)) pointNumbers.push(i); // both the time and expocode have to match.
            }
    
            if (pointNumbers.length) {
                // Trigger hover on the matching point(s)
                Plotly.Fx.hover(
                    mapPlot, 
                    pointNumbers.map(pn => ({curveNumber, pointNumber: pn})),
                    "geo"
                    );
            } else {
                // If no match, clear hover
                Plotly.Fx.unhover(mapPlot, "geo");
            }
    
            return t; // store last-hovered time_str (optional)
        }
    }
});