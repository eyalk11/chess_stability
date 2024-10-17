chrome.runtime.onInstalled.addListener(() => {
    console.log('Lichess Extension installed');
});

let socket;

function connectWebSocket() {
    socket = new WebSocket('ws://localhost:8765');

    socket.onopen = function (event) {
        console.log('WebSocket connection established');
        ///socket.send('Hello from Lichess Extension');
    };

    socket.onmessage = function (event) {
        console.log('Message from server:', event.data);
        // Forward the message to the content script
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, {
                    type: 'FROM_SERVER',
                    payload: event.data
                });
            }
        });
    };

    socket.onclose = function (event) {
        console.log('WebSocket connection closed');
        // Attempt to reconnect after a delay
        setTimeout(connectWebSocket, 50000);
    };

    socket.onerror = function (error) {
        console.error('WebSocket error:', error);
    };
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url.includes('lichess.org')) {
        chrome.tabs.sendMessage(tabId, { message: 'lichess_page_loaded' });
        //connectWebSocket();
    }
});

// Add this new message listener
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Received message in background script:', message);
    if (message.type === 'FROM_CONTENT') {
        console.log('Received message from content script:', message.payload);
        // Check if the WebSocket is not open, and if so, establish the connection
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            console.log('WebSocket not open. Establishing connection...');
            connectWebSocket();
        }

        // If the WebSocket is open, send the message to the server
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: 'FROM_CONTENT',
                payload: message.payload
            }));
        }
    }
});
