// filepath: marketing-dashboard/marketing-dashboard/static/js/chart_handlers.js

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('chart1-btn').addEventListener('click', function() {
        fetch('/chart1')
            .then(response => response.json())
            .then(data => {
                const chartContainer = document.getElementById('chart1-container');
                chartContainer.innerHTML = ''; // Clear previous chart
                const chart = document.createElement('div');
                chart.innerHTML = data.chart_html; // Assuming the backend returns HTML for the chart
                chartContainer.appendChild(chart);
            });
    });

    document.getElementById('chart2-btn').addEventListener('click', function() {
        fetch('/chart2')
            .then(response => response.json())
            .then(data => {
                const chartContainer = document.getElementById('chart2-container');
                chartContainer.innerHTML = ''; // Clear previous chart
                const chart = document.createElement('div');
                chart.innerHTML = data.chart_html; // Assuming the backend returns HTML for the chart
                chartContainer.appendChild(chart);
            });
    });

    document.getElementById('chart3-btn').addEventListener('click', function() {
        fetch('/chart3')
            .then(response => response.json())
            .then(data => {
                const chartContainer = document.getElementById('chart3-container');
                chartContainer.innerHTML = ''; // Clear previous chart
                const chart = document.createElement('div');
                chart.innerHTML = data.chart_html; // Assuming the backend returns HTML for the chart
                chartContainer.appendChild(chart);
            });
    });

    // Add more event listeners for additional charts as needed
});