console.log('Lichess Extension content script loaded');

// var arrive = document.createElement('script');
// arrive.src = chrome.extension.getURL('arrive.min.js');
// (document.head || document.documentElement).appendChild(arrive);
// arrive.onload = function () {
//     arrive.remove();
// };
var s = document.createElement('script');
s.src = chrome.extension.getURL('script.js');
(document.head || document.documentElement).appendChild(s);
s.onload = function () {
    s.remove();
};

// document.arrive("move", { fireOnAttributesModification: true, existing: true }, function () {
//     console.log('Arrived at analyse__moves');
//     document.dispatchEvent(new CustomEvent('get_current_move', {
//         detail: "aaa" // Some variable from Gmail.
//     }));

// });
// }, 3000); // 3000 milliseconds (3 seconds) delay
//console.log(site);


// Event listener
document.addEventListener('RW759_connectExtension', function (e) {
    // e.detail contains the transferred data (can be anything, ranging
    // from JavaScript objects to strings).
    // Do something, for example:
    console.log('RW759_connectExtension event received:', e.detail);
    //alert(e.detail);
    chrome.runtime.sendMessage({
        type: 'FROM_CONTENT', payload: e.detail
    })
});

// Function to create or update the custom analysis div
function createOrUpdateCustomDiv(text) {
    // Select the aside element
    var asideElement = document.querySelector('aside');

    // Check if the custom div already exists
    var customDiv = document.querySelector('.custom-analysis');

    if (!customDiv) {
        // If it doesn't exist, create a new div element
        customDiv = document.createElement('div');
        customDiv.className = 'custom-analysis';
        asideElement.appendChild(customDiv);
    }

    // Update the text content of the div
    customDiv.innerHTML = text;
}


chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.message === 'lichess_page_loaded') {
        console.log('Lichess page loaded, ready to interact');
        // Send a message to the background script
        // Check if site and site.analysis exist before accessing properties
        // if (site && site.analysis) {
        //     // Proceed with sending the message
        //     chrome.runtime.sendMessage({
        //         type: 'FROM_CONTENT', payload:
        //         {
        //             'fen': site.analysis.node,
        //             'nodes': site.analysis.nodeList
        //         }
        //     });
        // }
    } else if (request.type === 'FROM_SERVER') {
        console.log('FROM_SERVER event received:', request.payload);
        if (request.payload) {
            createOrUpdateCustomDiv(request.payload);
        } else {
            console.log('No client_text received WTF');
        }
    }
});


// You can add more functionality to interact with Lichess here
// For example, you could add event listeners to chess moves,
// modify the UI, or add custom buttons
