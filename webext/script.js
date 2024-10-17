console.log(site);
if (typeof site !== 'undefined' && site.analysis) {
    var tmp = site.analysis.onChange
    site.analysis.onChange = function () {
        console.log(typeof tmp)
        tmp()
        console.log('Site and analysis objects are available');
        console.log('Current FEN:', site.analysis.node.fen);
        console.log('Current path:', site.analysis.path);

        // Dispatch an event with the current move information
        document.dispatchEvent(new CustomEvent('RW759_connectExtension', {
            detail: {
                current: {
                    ply: site.analysis.node.ply,
                    fen: site.analysis.node.fen,
                    id: site.analysis.node.id,
                    uci: site.analysis.node.uci,
                    san: site.analysis.node.san
                },
                nodeList: site.analysis.nodeList.map(node => ({
                    ply: node.ply,
                    fen: node.fen,
                    id: node.id,
                    uci: node.uci,
                    san: node.san
                }))
            }
        }));
    }

} else {
    console.log('Site or analysis object is not available');
}


/*document.addEventListener('get_current_move', function (e) {
    if (typeof site !== 'undefined') {
        console.log('site object is available:', site);
        //         // You can add your logic here to interact with the site object
    } else {
        console.log('site object is not available after delay');

    }
});*/