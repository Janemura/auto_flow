async function fetchTrafficData() {
    try {
        const response = await fetch('/traffic_data');
        const data = await response.json();
        
        // Update the traffic data on the page
        document.getElementById('lane1_cars').textContent = data.lane1_cars;
        document.getElementById('lane2_cars').textContent = data.lane2_cars;
        document.getElementById('signal_status').textContent = data.signal_status;
        document.getElementById('route_recommendation').textContent = data.route_recommendation;
        
    } catch (error) {
        console.error('Error fetching traffic data:', 5000);
    }
}

// Function to handle evaluation button click
async function evaluateDetection() {
    try {
        const response = await fetch('/evaluate_detection', { method: 'POST' });
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            document.getElementById('precision').textContent = data.precision;
            document.getElementById('recall').textContent = data.recall;
            document.getElementById('f1').textContent = data.f1_score;
            alert('Detection evaluation complete!');
        }
    } catch (error) {
        console.error('Error during evaluation:', error);
        alert('Error during evaluation. See console for details.');
    }
}

// Initialize the page
function initPage() {
    // Add event listener to evaluation button
    const evaluateBtn = document.getElementById('evaluate-btn');
    if (evaluateBtn) {
        evaluateBtn.addEventListener('click', evaluateDetection);
    }
    
    // Fetch traffic data every 5 seconds
    setInterval(fetchTrafficData, 5000);
    
    // Initial fetch
    fetchTrafficData();
}

// Run initialization when DOM is fully loaded
document.addEventListener('DOMContentLoaded', initPage);