/**
 * MNIST Dataset Explorer JavaScript
 * Handles interactive functionality for the dataset exploration page
 */

// Global variables
let currentPage = 1;
let imagesPerPage = 50;
let totalImages = 70000; // Default value, will be updated from API
let selectedDataset = 'test';
let selectedDigits = []; // Store selected digit categories
let currentImageData = null; // Current image data being viewed

// DOM elements
document.addEventListener('DOMContentLoaded', function() {
    // Initialize datasets statistics and charts
    initializeDatasetStats();
    
    // Initialize filters
    initializeFilters();
    
    // Load the first page of images
    loadImages();
    
    // Initialize pagination
    initializePagination();
});

// Initialize datasets statistics and charts
function initializeDatasetStats() {
    // Fetch real statistics from the API
    fetch('/api/dataset_stats')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                const stats = data.stats;
                
                // Update statistics display
                document.getElementById('stat-total').textContent = stats.total_images.toLocaleString();
                document.getElementById('stat-train').textContent = stats.training_set.toLocaleString();
                document.getElementById('stat-test').textContent = stats.test_set.toLocaleString();
                document.getElementById('stat-size').textContent = stats.image_size;
                document.getElementById('stat-classes').textContent = stats.class_count;
                
                // Update global variable
                totalImages = selectedDataset === 'train' ? stats.training_set : stats.test_set;
                
                // Create distribution chart with real data
                createDistributionChart(stats.train_distribution, stats.test_distribution);
            } else {
                console.error('Failed to fetch dataset statistics:', data.message);
                // Fallback to placeholder message
                document.getElementById('stats-container').innerHTML = '<p class="text-center">Failed to load dataset statistics</p>';
            }
        })
        .catch(error => {
            console.error('Error fetching dataset statistics:', error);
            document.getElementById('stats-container').innerHTML = '<p class="text-center">Failed to load dataset statistics</p>';
        });
}

// Create distribution chart
function createDistributionChart(trainData, testData) {
    const ctx = document.getElementById('distribution-chart').getContext('2d');
    
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            datasets: [
                {
                    label: 'Training Set',
                    data: trainData,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Test Set',
                    data: testData,
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Images'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Digit Class'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'MNIST Dataset Distribution by Digit Class'
                },
                legend: {
                    position: 'top'
                }
            }
        }
    });
}

// Initialize filters
function initializeFilters() {
    // Get digit filter and dataset selector
    const digitFilter = document.getElementById('digit-filter');
    const datasetFilter = document.getElementById('dataset-filter');
    
    // Listen for digit filter change
    digitFilter.addEventListener('change', function() {
        const selectedDigit = parseInt(this.value);
        
        if(selectedDigit === -1) {
            // Select "All Digits" option
            selectedDigits = [];
        } else if(!selectedDigits.includes(selectedDigit)) {
            // Add to selected digit array
            selectedDigits.push(selectedDigit);
        }
        
        // Reset to first page and load images
        currentPage = 1;
        loadImages();
    });
    
    // Listen for dataset selector change
    datasetFilter.addEventListener('change', function() {
        selectedDataset = this.value;
        // Update total images based on selected dataset
        fetch('/api/dataset_stats')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    totalImages = selectedDataset === 'train' ? 
                        data.stats.training_set : data.stats.test_set;
                    currentPage = 1;
                    updatePagination();
                    loadImages();
                }
            });
    });
    
    // Add clear filters button event
    document.getElementById('clear-filters').addEventListener('click', function() {
        selectedDigits = [];
        digitFilter.value = -1;
        currentPage = 1;
        loadImages();
    });
}

// Initialize pagination
function initializePagination() {
    // Previous page button
    document.getElementById('prev-page').addEventListener('click', function() {
        if(currentPage > 1) {
            currentPage--;
            loadImages();
        }
    });
    
    // Next page button
    document.getElementById('next-page').addEventListener('click', function() {
        const maxPage = Math.ceil(totalImages / imagesPerPage);
        if(currentPage < maxPage) {
            currentPage++;
            loadImages();
        }
    });
    
    // Update pagination display
    updatePagination();
}

// Update pagination information
function updatePagination() {
    const maxPage = Math.ceil(totalImages / imagesPerPage);
    
    // Update page number display
    document.getElementById('current-page').textContent = currentPage;
    document.getElementById('total-pages').textContent = maxPage;
    
    // Enable/disable previous and next buttons
    document.getElementById('prev-page').disabled = (currentPage === 1);
    document.getElementById('next-page').disabled = (currentPage === maxPage);
}

// Load images from API
function loadImages() {
    const imageGrid = document.getElementById('image-grid');
    
    // Clear image grid
    imageGrid.innerHTML = '<div class="text-center p-5"><div class="spinner-border" role="status"></div><p class="mt-2">Loading images...</p></div>';
    
    // Prepare request data
    const requestData = {
        dataset: selectedDataset,
        count: imagesPerPage
    };
    
    // Add digit filter if any digit is selected
    if (selectedDigits.length === 1) {
        requestData.digit = selectedDigits[0];
    }
    
    // Fetch real images from the API
    fetch('/api/dataset_samples', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success' && data.samples) {
            // Clear image grid
            imageGrid.innerHTML = '';
            
            // Render each sample
            data.samples.forEach(sample => {
                const card = document.createElement('div');
                card.className = 'image-card';
                
                const img = document.createElement('img');
                img.src = `data:image/png;base64,${sample.image}`;
                img.alt = `Digit ${sample.true_label}`;
                
                const label = document.createElement('div');
                label.className = 'image-label';
                label.textContent = `${sample.true_label}`;
                
                card.appendChild(img);
                card.appendChild(label);
                imageGrid.appendChild(card);
                
                // Click handler to show larger image
                card.addEventListener('click', () => {
                    showImageModal(sample);
                });
            });
            
            // Update total statistics
            updatePagination(data.filtered_count);
        } else {
            imageGrid.innerHTML = `<div class="alert alert-warning">No images found.</div>`;
        }
    })
    .catch(error => {
        console.error('Error loading images:', error);
        imageGrid.innerHTML = `<div class="alert alert-danger">Error loading images: ${error.message}</div>`;
    });
}

// Show image in modal
function showImageModal(sample) {
    // Create modal elements
    const modal = document.createElement('div');
    modal.className = 'modal fade show';
    modal.style.display = 'block';
    modal.tabIndex = '-1';
    
    const dialog = document.createElement('div');
    dialog.className = 'modal-dialog modal-dialog-centered';
    
    const content = document.createElement('div');
    content.className = 'modal-content';
    
    const header = document.createElement('div');
    header.className = 'modal-header';
    header.innerHTML = `<h5 class="modal-title">Digit ${sample.true_label}</h5>`;
    
    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'btn-close';
    closeButton.setAttribute('data-bs-dismiss', 'modal');
    closeButton.addEventListener('click', () => {
        document.body.removeChild(modal);
        document.body.classList.remove('modal-open');
    });
    
    const body = document.createElement('div');
    body.className = 'modal-body text-center';
    
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${sample.image}`;
    img.alt = `Digit ${sample.true_label}`;
    img.className = 'img-fluid modal-image';
    
    const info = document.createElement('div');
    info.className = 'mt-3';
    info.innerHTML = `
        <p><strong>Dataset:</strong> ${selectedDataset}</p>
            <p><strong>Index:</strong> ${sample.index}</p>
    `;
    
    // Assemble modal
    header.appendChild(closeButton);
    body.appendChild(img);
    body.appendChild(info);
    
    content.appendChild(header);
    content.appendChild(body);
    dialog.appendChild(content);
    modal.appendChild(dialog);
    
    // Add backdrop
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop fade show';
    
    // Add to body
    document.body.appendChild(modal);
    document.body.appendChild(backdrop);
    document.body.classList.add('modal-open');
    
    // Add click handler to close modal when clicking outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            document.body.removeChild(modal);
            document.body.removeChild(backdrop);
            document.body.classList.remove('modal-open');
        }
    });
}

// Create pixel intensity heatmap
function createHeatmap(pixels) {
    const heatmapDiv = document.getElementById('pixel-heatmap');
    heatmapDiv.innerHTML = '';
    
    // Create 28x28 grid
    for(let i = 0; i < pixels.length; i++) {
        const cell = document.createElement('div');
        cell.className = 'heatmap-cell';
        
        // Set background color based on pixel value
        const value = pixels[i];
        const intensity = Math.floor(value); // 0-255
        
        cell.style.backgroundColor = `rgb(${intensity}, ${intensity}, ${intensity})`;
        
        // Add to heatmap
        heatmapDiv.appendChild(cell);
    }
}

// Export the current image as a PNG file
function exportImageToPNG() {
    if (!currentImageData) return;
    
    // Create a canvas element
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    
    // Create an Image element to load the base64 data
    const img = new Image();
    img.onload = function() {
        // Draw the image to the canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        
        // Convert canvas to data URL and trigger download
        const dataURL = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.href = dataURL;
        link.download = `mnist-digit-${currentImageData.label}-${currentImageData.index}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };
    
    // Set the source to the base64 image
    img.src = `data:image/png;base64,${currentImageData.image}`;
}

// Draw image data on canvas
function drawImageData(canvas, pixels) {
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(28, 28);
    
    for(let i = 0; i < pixels.length; i++) {
        const value = pixels[i];
        imgData.data[i * 4] = value;     // R
        imgData.data[i * 4 + 1] = value; // G
        imgData.data[i * 4 + 2] = value; // B
        imgData.data[i * 4 + 3] = 255;   // A
    }
    
    ctx.putImageData(imgData, 0, 0);
} 