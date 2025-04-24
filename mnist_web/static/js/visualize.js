/**
 * MNIST Visualization JavaScript
 * Handles interactive visualizations for MNIST dataset and models
 */

// Global state and configuration
const state = {
    currentModel: null,
    availableModels: [],
    visualizationType: 'confusion-matrix',
    dataset: 'test',
    digit: 'all',
    sampleSize: 100,
    currentVisualization: null,
    currentData: null
};

// DOM elements
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the visualizations page
    initializeVisualizationOptions();
    setupEventListeners();
    loadAvailableModels();

    // Update UI based on initial state
    updateVisibleOptions();
});

/**
 * Initialize visualization options and interface
 */
function initializeVisualizationOptions() {
    // Set up sample size range slider value display
    const sampleSizeInput = document.getElementById('sample-size');
    const sampleSizeValue = document.getElementById('sample-size-value');
    sampleSizeValue.textContent = sampleSizeInput.value;
    
    sampleSizeInput.addEventListener('input', function() {
        sampleSizeValue.textContent = this.value;
        state.sampleSize = parseInt(this.value);
    });
}

/**
 * Set up event listeners for UI interactions
 */
function setupEventListeners() {
    // Visualization Type selector
    document.getElementById('visualization-type').addEventListener('change', function() {
        state.visualizationType = this.value;
        updateVisibleOptions();
    });
    
    // Model selector
    document.getElementById('model-select').addEventListener('change', function() {
        state.currentModel = this.value;
    });
    
    // Dataset selector
    document.getElementById('dataset-select').addEventListener('change', function() {
        state.dataset = this.value;
    });
    
    // Digit selector
    document.getElementById('digit-select').addEventListener('change', function() {
        state.digit = this.value;
    });
    
    // Generate button - fix the button ID to match HTML
    document.getElementById('generate-btn').addEventListener('click', generateVisualization);
    
    // Reset button - fix the button ID to match HTML
    document.getElementById('reset-options-btn').addEventListener('click', resetOptions);
    
    // Export buttons
    document.getElementById('export-image-btn').addEventListener('click', exportAsImage);
    document.getElementById('export-data-btn').addEventListener('click', exportData);
}

/**
 * Load available models from the server
 */
function loadAvailableModels() {
    showLoading(true);
    
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                state.availableModels = data.models;
                populateModelSelector(data.models);
            } else {
                showError('Failed to load models: ' + data.message);
            }
            showLoading(false);
        })
        .catch(error => {
            console.error('Error loading models:', error);
            showError('Failed to load models. Check console for details.');
            showLoading(false);
        });
}

/**
 * Populate model selector dropdown with available models
 */
function populateModelSelector(models) {
    const modelSelect = document.getElementById('model-select');
    
    // Clear existing options
    modelSelect.innerHTML = '';
    
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = 'current';
    defaultOption.textContent = 'Current Model';
    modelSelect.appendChild(defaultOption);
    
    // Add models to the selector
    if (models && Object.keys(models).length > 0) {
        for (const [id, model] of Object.entries(models)) {
            const option = document.createElement('option');
            option.value = id;
            // 使用性能对象中的准确度数据，而不是假设有best_accuracy字段
            const accuracy = model.performance ? model.performance.accuracy : 0;
            option.textContent = `${model.name || id} (Acc: ${(accuracy * 100).toFixed(2)}%)`;
            modelSelect.appendChild(option);
        }
    } else {
        const noModelOption = document.createElement('option');
        noModelOption.value = '';
        noModelOption.textContent = 'No trained models available';
        noModelOption.disabled = true;
        modelSelect.appendChild(noModelOption);
    }
    
    // Set current model as default
    state.currentModel = 'current';
}

/**
 * Update which options are visible based on visualization type
 */
function updateVisibleOptions() {
    const digitContainer = document.getElementById('digit-select-container');
    const sampleSizeContainer = document.getElementById('sample-size-container');
    
    // Show/hide options based on visualization type
    switch (state.visualizationType) {
        case 'confusion-matrix':
            digitContainer.style.display = 'none';
            sampleSizeContainer.style.display = 'none';
            break;
        case 'digit-samples':
            digitContainer.style.display = 'block';
            sampleSizeContainer.style.display = 'block';
            break;
        case 'misclassified':
            digitContainer.style.display = 'block';
            sampleSizeContainer.style.display = 'block';
            break;
        case 'prediction-confidence':
            digitContainer.style.display = 'block';
            sampleSizeContainer.style.display = 'block';
            break;
        case 'tsne':
            digitContainer.style.display = 'none';
            sampleSizeContainer.style.display = 'block';
            break;
    }
}

/**
 * Reset visualization options to defaults
 */
function resetOptions() {
    // Reset form elements
    document.getElementById('visualization-type').value = 'confusion-matrix';
    if (document.getElementById('model-select').options.length > 0) {
        document.getElementById('model-select').selectedIndex = 0;
    }
    document.getElementById('dataset-select').value = 'test';
    document.getElementById('digit-select').value = 'all';
    document.getElementById('sample-size').value = 100;
    document.getElementById('sample-size-value').textContent = '100';
    
    // Reset state
    state.visualizationType = 'confusion-matrix';
    state.currentModel = 'current';
    state.dataset = 'test';
    state.digit = 'all';
    state.sampleSize = 100;
    
    // Update UI
    updateVisibleOptions();
    
    // Clear visualization
    document.getElementById('visualization-placeholder').style.display = 'block';
    document.getElementById('visualization-container').style.display = 'none';
    document.querySelector('.insights-section').style.display = 'none';
    document.getElementById('data-section').style.display = 'none';
}

/**
 * Generate visualization based on selected options
 */
function generateVisualization() {
    // Hide placeholder and insights
    document.getElementById('visualization-placeholder').style.display = 'none';
    document.getElementById('visualization-container').style.display = 'block';
    document.querySelector('.insights-section').style.display = 'none';
    document.getElementById('data-section').style.display = 'none';
    
    // Clear visualization content
    document.getElementById('visualization-content').innerHTML = '';
    document.getElementById('visualization-caption').innerHTML = '';
    
    // Show loading indicator
    showLoading(true);
    
    // Set visualization title
    document.getElementById('visualization-title').textContent = 
        `${state.visualizationType.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())} - ${state.dataset} Dataset`;
    
    // Generate the appropriate visualization based on type
    switch (state.visualizationType) {
        case 'confusion-matrix':
            generateConfusionMatrix();
            break;
        case 'digit-samples':
            generateDigitSamples();
            break;
        case 'prediction-confidence':
            generatePredictionConfidence();
            break;
        case 'tsne':
            generateTSNE();
            break;
        case 'misclassified':
            generateMisclassifiedExamples();
            break;
        case 'model-weights':
            generateModelWeights();
            break;
        default:
            showError('Unsupported visualization type');
            showLoading(false);
    }
}

/**
 * Generate confusion matrix visualization
 */
function generateConfusionMatrix() {
    // Prepare request data
    const requestData = {
        model_id: state.currentModel === 'current' ? null : state.currentModel,
        dataset: state.dataset
    };
    
    // Make API request
    fetch('/api/visualize/confusion_matrix', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            state.currentData = data;
            renderConfusionMatrix(data.confusion_matrix, data.accuracy);
            
            // Show insights
            document.querySelector('.insights-section').style.display = 'block';
            generateInsights(data);
            
            // Show data table
            document.getElementById('data-section').style.display = 'block';
            showDataTab(data);
        } else {
            showError('Failed to generate confusion matrix: ' + data.message);
        }
        showLoading(false);
    })
    .catch(error => {
        console.error('Error generating confusion matrix:', error);
        showError('Failed to generate confusion matrix. Check console for details.');
        showLoading(false);
    });
}

/**
 * Render confusion matrix visualization
 */
function renderConfusionMatrix(matrix, accuracy) {
    const container = document.getElementById('visualization-content');
    
    // Set up data for confusion matrix heatmap
    const data = [{
        z: matrix,
        x: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        y: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        type: 'heatmap',
        colorscale: 'Blues',
        showscale: true,
        hoverongaps: false,
        colorbar: {
            title: 'Count',
            thickness: 20,
        },
        hovertemplate: 'True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    }];
    
    // Layout configuration
    const layout = {
        title: {
            text: `Confusion Matrix (Accuracy: ${(accuracy * 100).toFixed(2)}%)`,
            font: { size: 18 }
        },
        xaxis: {
            title: 'Predicted Label',
            tickfont: { size: 14 }
        },
        yaxis: {
            title: 'True Label',
            tickfont: { size: 14 }
        },
        annotations: [],
        margin: { t: 80, r: 30, b: 60, l: 60 },
        height: 600,
        width: Math.min(700, window.innerWidth - 60)
    };
    
    // Add text annotations to the heatmap
    for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 10; j++) {
            const value = matrix[i][j];
            const textColor = value > (Math.max(...matrix.flat()) / 2) ? 'white' : 'black';
            
            layout.annotations.push({
                x: j,
                y: i,
                text: value.toString(),
                font: {
                    color: textColor
                },
                showarrow: false
            });
        }
    }
    
    // Create the plot
    Plotly.newPlot(container, data, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    });
    
    // Set caption
    document.getElementById('visualization-caption').innerHTML = `
        <p>This confusion matrix shows the distribution of predictions for each digit in the ${state.dataset} dataset. 
        The rows represent the true labels, and the columns represent the predicted labels. 
        The diagonal elements represent correctly classified instances, while off-diagonal elements represent misclassifications.</p>
        <p>Overall model accuracy: <strong>${(accuracy * 100).toFixed(2)}%</strong></p>
    `;
}

/**
 * Generate digit samples visualization
 */
function generateDigitSamples() {
    // Prepare request data
    const requestData = {
        dataset: state.dataset,
        digit: state.digit === 'all' ? null : parseInt(state.digit),
        count: state.sampleSize
    };
    
    // Make API request
    fetch('/api/dataset_samples', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            state.currentData = data;
            renderDigitSamples(data.samples);
            
            // Show insights
            document.querySelector('.insights-section').style.display = 'block';
            generateInsights(data);
            
            // Show data
            document.getElementById('data-section').style.display = 'block';
            showDataTab(data);
        } else {
            showError('Failed to get digit samples: ' + data.message);
        }
        showLoading(false);
    })
    .catch(error => {
        console.error('Error getting digit samples:', error);
        showError('Failed to get digit samples. Check console for details.');
        showLoading(false);
    });
}

/**
 * Render digit samples visualization
 */
function renderDigitSamples(samples) {
    const container = document.getElementById('visualization-content');
    container.innerHTML = '';
    
    // Create grid for digit samples
    const grid = document.createElement('div');
    grid.className = 'digit-samples-grid';
    container.appendChild(grid);
    
    // Add each sample to the grid
    samples.forEach(sample => {
        const sampleDiv = document.createElement('div');
        sampleDiv.className = 'digit-sample';
        
        // Create image element
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${sample.image}`;
        img.alt = `Digit ${sample.true_label}`;
        img.title = `Digit ${sample.true_label} (Index: ${sample.index})`;
        
        // Add label below the image
        const label = document.createElement('div');
        label.className = 'digit-label';
        label.textContent = `${sample.true_label}`;
        
        sampleDiv.appendChild(img);
        sampleDiv.appendChild(label);
        grid.appendChild(sampleDiv);
        
        // Add click handler to show larger image
        sampleDiv.addEventListener('click', () => {
            showLargeDigitModal(sample);
        });
    });
    
    // Set caption
    let captionText = `Showing ${samples.length} random samples from the ${state.dataset} dataset`;
    if (state.digit !== 'all') {
        captionText = `Showing ${samples.length} random samples of digit ${state.digit} from the ${state.dataset} dataset`;
    }
    
    document.getElementById('visualization-caption').innerHTML = `<p>${captionText}</p>`;
}

/**
 * Show a modal with a larger version of the digit
 */
function showLargeDigitModal(sample) {
    // Create modal container
    const modal = document.createElement('div');
    modal.className = 'large-digit-modal';
    modal.style.position = 'fixed';
    modal.style.top = '0';
    modal.style.left = '0';
    modal.style.width = '100%';
    modal.style.height = '100%';
    modal.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    modal.style.display = 'flex';
    modal.style.justifyContent = 'center';
    modal.style.alignItems = 'center';
    modal.style.zIndex = '1000';
    
    // Create modal content
    const content = document.createElement('div');
    content.style.backgroundColor = 'white';
    content.style.padding = '20px';
    content.style.borderRadius = '8px';
    content.style.maxWidth = '400px';
    content.style.textAlign = 'center';
    
    // Create image
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${sample.image}`;
    img.alt = `Digit ${sample.true_label}`;
    img.style.width = '200px';
    img.style.height = '200px';
    img.style.objectFit = 'contain';
    
    // Create info text
    const info = document.createElement('div');
    info.innerHTML = `
        <h3>Digit ${sample.true_label}</h3>
        <p>Dataset: ${state.dataset}</p>
        <p>Index: ${sample.index}</p>
    `;
    
    // Create close button
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    closeBtn.className = 'btn btn-primary';
    closeBtn.style.marginTop = '15px';
    closeBtn.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    // Assemble modal
    content.appendChild(img);
    content.appendChild(info);
    content.appendChild(closeBtn);
    modal.appendChild(content);
    
    // Add click handler to close when clicking outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            document.body.removeChild(modal);
        }
    });
    
    // Add to document
    document.body.appendChild(modal);
}

/**
 * Generate prediction confidence visualization
 */
function generatePredictionConfidence() {
    // Prepare request data
    const requestData = {
        model_id: state.currentModel === 'current' ? null : state.currentModel,
        dataset: state.dataset,
        digit: state.digit === 'all' ? null : parseInt(state.digit),
        count: state.sampleSize
    };
    
    // Make API request
    fetch('/api/visualize/prediction_confidence', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            state.currentData = data;
            renderPredictionConfidence(data.predictions);
            
            // Show insights
            document.querySelector('.insights-section').style.display = 'block';
            generateInsights(data);
            
            // Show data
            document.getElementById('data-section').style.display = 'block';
            showDataTab(data);
        } else {
            showError('Failed to generate prediction confidence visualization: ' + data.message);
        }
        showLoading(false);
    })
    .catch(error => {
        console.error('Error generating prediction confidence:', error);
        showError('Failed to generate prediction confidence. Check console for details.');
        showLoading(false);
    });
}

/**
 * Render prediction confidence visualization
 */
function renderPredictionConfidence(predictions) {
    const container = document.getElementById('visualization-content');
    
    // Calculate confidence distribution for each digit
    const confidenceByDigit = Array(10).fill().map(() => []);
    
    // Group confidences by true digit
    predictions.forEach(pred => {
        const trueDigit = pred.true_label;
        const predDigit = pred.predicted_label;
        const confidence = pred.confidence;
        
        if (trueDigit === predDigit) {
            confidenceByDigit[trueDigit].push(confidence);
        }
    });
    
    // Calculate statistics for each digit
    const stats = confidenceByDigit.map((confidences, digit) => {
        if (confidences.length === 0) return { digit, count: 0, avg: 0, median: 0 };
        
        const sorted = [...confidences].sort((a, b) => a - b);
        const sum = confidences.reduce((a, b) => a + b, 0);
        const avg = sum / confidences.length;
        const median = sorted[Math.floor(sorted.length / 2)];
        
        return {
            digit,
            count: confidences.length,
            avg,
            median,
            data: confidences
        };
    });
    
    // Set up data for box plot
    const boxPlotData = stats.map(stat => {
        return {
            y: stat.data || [],
            type: 'box',
            name: stat.digit.toString(),
            boxmean: true,
            jitter: 0.3,
            pointpos: 0,
            boxpoints: 'all'
        };
    });
    
    // Filter out digits with no data
    const filteredBoxPlotData = boxPlotData.filter(d => d.y.length > 0);
    
    // Layout configuration
    const layout = {
        title: {
            text: 'Prediction Confidence by Digit',
            font: { size: 18 }
        },
        xaxis: {
            title: 'Digit',
            tickfont: { size: 14 }
        },
        yaxis: {
            title: 'Confidence Score',
            tickfont: { size: 14 },
            range: [0, 1]
        },
        margin: { t: 80, r: 30, b: 60, l: 60 },
        height: 600,
        width: Math.min(700, window.innerWidth - 60)
    };
    
    // Create the plot
    Plotly.newPlot(container, filteredBoxPlotData, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    });
    
    // Set caption
    document.getElementById('visualization-caption').innerHTML = `
        <p>This box plot shows the distribution of prediction confidence scores for correctly classified digits.
        Each box represents the distribution of confidence values for predictions of a single digit.</p>
        <p>The box shows the interquartile range (IQR), with the line inside the box representing the median confidence.
        Points outside the whiskers represent potential outliers.</p>
    `;
}

/**
 * Generate t-SNE visualization
 */
function generateTSNE() {
    // Prepare request data
    const requestData = {
        model_id: state.currentModel === 'current' ? null : state.currentModel,
        dataset: state.dataset,
        sample_size: state.sampleSize
    };
    
    // Make API request
    fetch('/api/visualize/tsne', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            state.currentData = data;
            renderTSNE(data.tsne_results);
            
            // Show insights
            document.querySelector('.insights-section').style.display = 'block';
            generateInsights(data);
            
            // Show data
            document.getElementById('data-section').style.display = 'block';
            showDataTab(data);
        } else {
            showError('Failed to generate t-SNE visualization: ' + data.message);
        }
        showLoading(false);
    })
    .catch(error => {
        console.error('Error generating t-SNE:', error);
        showError('Failed to generate t-SNE visualization. Check console for details.');
        showLoading(false);
    });
}

/**
 * Render t-SNE visualization
 */
function renderTSNE(results) {
    const container = document.getElementById('visualization-content');
    
    // Separate data by digit
    const dataByDigit = Array(10).fill().map(() => ({ x: [], y: [], text: [] }));
    
    results.forEach(point => {
        const digit = point.label;
        dataByDigit[digit].x.push(point.x);
        dataByDigit[digit].y.push(point.y);
        dataByDigit[digit].text.push(`Digit: ${digit}<br>Index: ${point.index}`);
    });
    
    // Create traces for each digit
    const traces = dataByDigit.map((data, digit) => {
        return {
            x: data.x,
            y: data.y,
            mode: 'markers',
            type: 'scatter',
            name: `Digit ${digit}`,
            text: data.text,
            hovertemplate: '%{text}<extra></extra>',
            marker: {
                size: 8
            }
        };
    });
    
    // Filter out digits with no data
    const filteredTraces = traces.filter(trace => trace.x.length > 0);
    
    // Layout configuration
    const layout = {
        title: {
            text: 't-SNE Visualization of MNIST Digits',
            font: { size: 18 }
        },
        xaxis: {
            title: 't-SNE Dimension 1',
            zeroline: false
        },
        yaxis: {
            title: 't-SNE Dimension 2',
            zeroline: false
        },
        margin: { t: 80, r: 30, b: 60, l: 60 },
        height: 600,
        width: Math.min(700, window.innerWidth - 60),
        hovermode: 'closest',
        legend: {
            x: 1,
            y: 0.5
        }
    };
    
    // Create the plot
    Plotly.newPlot(container, filteredTraces, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    });
    
    // Set caption
    document.getElementById('visualization-caption').innerHTML = `
        <p>This t-SNE visualization shows the MNIST digits projected into a 2D space while preserving local similarities.
        Each point represents a single digit from the ${state.dataset} dataset, colored by its true label.</p>
        <p>t-SNE (t-Distributed Stochastic Neighbor Embedding) helps visualize high-dimensional data by giving each 
        datapoint a location in a 2D or 3D map, revealing clusters of similar points and the overall structure of the data.</p>
    `;
}

/**
 * Generate misclassified examples visualization
 */
function generateMisclassifiedExamples() {
    // Prepare request data
    const requestData = {
        model_id: state.currentModel === 'current' ? null : state.currentModel,
        dataset: state.dataset,
        digit: state.digit === 'all' ? null : parseInt(state.digit),
        count: state.sampleSize
    };
    
    // Make API request
    fetch('/api/visualize/misclassified', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            state.currentData = data;
            renderMisclassifiedExamples(data.misclassified);
            
            // Show insights
            document.querySelector('.insights-section').style.display = 'block';
            generateInsights(data);
            
            // Show data
            document.getElementById('data-section').style.display = 'block';
            showDataTab(data);
        } else {
            showError('Failed to generate misclassified examples: ' + data.message);
        }
        showLoading(false);
    })
    .catch(error => {
        console.error('Error generating misclassified examples:', error);
        showError('Failed to generate misclassified examples. Check console for details.');
        showLoading(false);
    });
}

/**
 * Render misclassified examples visualization
 */
function renderMisclassifiedExamples(misclassified) {
    const container = document.getElementById('visualization-content');
    container.innerHTML = '';
    
    if (misclassified.length === 0) {
        container.innerHTML = `<div class="no-results">No misclassified examples found for the selected criteria.</div>`;
        return;
    }
    
    // Create grid for misclassified examples
    const grid = document.createElement('div');
    grid.className = 'digit-samples-grid';
    container.appendChild(grid);
    
    // Add each misclassified example to the grid
    misclassified.forEach(example => {
        const sampleDiv = document.createElement('div');
        sampleDiv.className = 'digit-sample misclassified';
        
        // Create image element
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${example.image}`;
        img.alt = `True: ${example.true_label}, Pred: ${example.predicted_label}`;
        img.title = `True: ${example.true_label}, Predicted: ${example.predicted_label} (Conf: ${(example.confidence * 100).toFixed(1)}%)`;
        
        // Add labels below the image
        const labelDiv = document.createElement('div');
        labelDiv.className = 'misclassified-labels';
        
        const trueLabel = document.createElement('span');
        trueLabel.className = 'true-label';
        trueLabel.textContent = `True: ${example.true_label}`;
        
        const predLabel = document.createElement('span');
        predLabel.className = 'pred-label';
        predLabel.textContent = `Pred: ${example.predicted_label}`;
        
        labelDiv.appendChild(trueLabel);
        labelDiv.appendChild(predLabel);
        
        sampleDiv.appendChild(img);
        sampleDiv.appendChild(labelDiv);
        grid.appendChild(sampleDiv);
        
        // Add click handler to show larger image with details
        sampleDiv.addEventListener('click', () => {
            showMisclassifiedModal(example);
        });
    });
    
    // Set caption
    let captionText = `Showing ${misclassified.length} misclassified examples from the ${state.dataset} dataset`;
    if (state.digit !== 'all') {
        captionText = `Showing ${misclassified.length} misclassified examples of digit ${state.digit} from the ${state.dataset} dataset`;
    }
    
    document.getElementById('visualization-caption').innerHTML = `<p>${captionText}</p>`;
}

/**
 * Show a modal with details about a misclassified example
 */
function showMisclassifiedModal(example) {
    // Create modal container
    const modal = document.createElement('div');
    modal.className = 'large-digit-modal';
    modal.style.position = 'fixed';
    modal.style.top = '0';
    modal.style.left = '0';
    modal.style.width = '100%';
    modal.style.height = '100%';
    modal.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    modal.style.display = 'flex';
    modal.style.justifyContent = 'center';
    modal.style.alignItems = 'center';
    modal.style.zIndex = '1000';
    
    // Create modal content
    const content = document.createElement('div');
    content.style.backgroundColor = 'white';
    content.style.padding = '20px';
    content.style.borderRadius = '8px';
    content.style.maxWidth = '400px';
    content.style.textAlign = 'center';
    
    // Create image
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${example.image}`;
    img.alt = `True: ${example.true_label}, Pred: ${example.predicted_label}`;
    img.style.width = '200px';
    img.style.height = '200px';
    img.style.objectFit = 'contain';
    
    // Create confidence bar chart if confidence scores are available
    let confidenceChart = '';
    if (example.all_confidences) {
        confidenceChart = document.createElement('div');
        confidenceChart.style.width = '100%';
        confidenceChart.style.height = '250px';
        confidenceChart.style.marginTop = '20px';
    }
    
    // Create info text
    const info = document.createElement('div');
    info.innerHTML = `
        <h3>Misclassified Digit</h3>
        <p><strong>True Label:</strong> ${example.true_label}</p>
        <p><strong>Predicted Label:</strong> ${example.predicted_label}</p>
        <p><strong>Confidence:</strong> ${(example.confidence * 100).toFixed(2)}%</p>
        <p><strong>Dataset:</strong> ${state.dataset}</p>
        <p><strong>Index:</strong> ${example.index}</p>
    `;
    
    // Create close button
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    closeBtn.className = 'btn btn-primary';
    closeBtn.style.marginTop = '15px';
    closeBtn.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    // Assemble modal
    content.appendChild(img);
    content.appendChild(info);
    if (confidenceChart) {
        content.appendChild(confidenceChart);
    }
    content.appendChild(closeBtn);
    modal.appendChild(content);
    
    // Add click handler to close when clicking outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            document.body.removeChild(modal);
        }
    });
    
    // Add to document
    document.body.appendChild(modal);
    
    // Render confidence chart if data is available
    if (example.all_confidences && confidenceChart) {
        const labels = Array.from({length: 10}, (_, i) => i.toString());
        const values = labels.map(label => {
            return example.all_confidences[label] || 0;
        });
        
        const data = [{
            x: labels,
            y: values,
            type: 'bar',
            marker: {
                color: labels.map(label => 
                    label === example.true_label.toString() ? '#4CAF50' : 
                    (label === example.predicted_label.toString() ? '#F44336' : '#2196F3')
                )
            }
        }];
        
        const layout = {
            title: 'Prediction Confidences',
            xaxis: { title: 'Digit' },
            yaxis: { 
                title: 'Confidence',
                range: [0, 1]
            },
            margin: { t: 50, r: 20, b: 50, l: 50 }
        };
        
        Plotly.newPlot(confidenceChart, data, layout);
    }
}

/**
 * Show data in tabular format
 */
function showDataTab(data) {
    const container = document.getElementById('data-container');
    container.innerHTML = '';
    
    if (!data) {
        container.innerHTML = '<p>No data available</p>';
        return;
    }
    
    switch (state.visualizationType) {
        case 'confusion-matrix':
            showConfusionMatrixData(data, container);
            break;
        case 'digit-samples':
            showDigitSamplesData(data, container);
            break;
        case 'prediction-confidence':
            showPredictionConfidenceData(data, container);
            break;
        case 'tsne':
            showTSNEData(data, container);
            break;
        case 'misclassified':
            showMisclassifiedData(data, container);
            break;
        default:
            container.innerHTML = '<p>No data available for this visualization type</p>';
    }
}

/**
 * Show confusion matrix data in tabular format
 */
function showConfusionMatrixData(data, container) {
    const table = document.createElement('table');
    table.className = 'data-table';
    
    // Create header row
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = '<th>True\\Pred</th>';
    
    for (let i = 0; i < 10; i++) {
        const th = document.createElement('th');
        th.textContent = i;
        headerRow.appendChild(th);
    }
    
    headerRow.innerHTML += '<th>Total</th><th>Accuracy</th>';
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body rows
    const tbody = document.createElement('tbody');
    
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('tr');
        const rowHeader = document.createElement('th');
        rowHeader.textContent = i;
        row.appendChild(rowHeader);
        
        let rowTotal = 0;
        let correctCount = 0;
        
        for (let j = 0; j < 10; j++) {
            const td = document.createElement('td');
            const count = data.confusion_matrix[i][j];
            td.textContent = count;
            
            // Highlight diagonal (correct predictions)
            if (i === j) {
                td.className = 'correct';
                correctCount = count;
            }
            
            rowTotal += count;
            row.appendChild(td);
        }
        
        // Add row total
        const totalTd = document.createElement('td');
        totalTd.textContent = rowTotal;
        totalTd.className = 'total';
        row.appendChild(totalTd);
        
        // Add row accuracy
        const accuracyTd = document.createElement('td');
        const accuracy = rowTotal > 0 ? (correctCount / rowTotal) * 100 : 0;
        accuracyTd.textContent = `${accuracy.toFixed(2)}%`;
        accuracyTd.className = 'accuracy';
        row.appendChild(accuracyTd);
        
        tbody.appendChild(row);
    }
    
    table.appendChild(tbody);
    container.appendChild(table);
    
    // Add overall accuracy
    const accuracyInfo = document.createElement('p');
    accuracyInfo.innerHTML = `<strong>Overall Accuracy:</strong> ${(data.accuracy * 100).toFixed(2)}%`;
    container.appendChild(accuracyInfo);
}

/**
 * Show digit samples data in tabular format
 */
function showDigitSamplesData(data, container) {
    if (!data.samples || data.samples.length === 0) {
        container.innerHTML = '<p>No sample data available</p>';
        return;
    }
    
    // Count samples per label
    const countByLabel = {};
    data.samples.forEach(sample => {
        const label = sample.true_label;
        countByLabel[label] = (countByLabel[label] || 0) + 1;
    });
    
    // Create statistics info
    const statsDiv = document.createElement('div');
    statsDiv.innerHTML = `
        <p><strong>Total samples:</strong> ${data.samples.length}</p>
        <p><strong>Dataset:</strong> ${data.dataset}</p>
        <p><strong>Filter:</strong> ${state.digit === 'all' ? 'All digits' : `Digit ${state.digit}`}</p>
    `;
    
    // Create count by digit table
    const table = document.createElement('table');
    table.className = 'data-table';
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = '<th>Digit</th><th>Count</th><th>Percentage</th>';
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    Object.entries(countByLabel).sort((a, b) => a[0] - b[0]).forEach(([label, count]) => {
        const row = document.createElement('tr');
        const percentage = (count / data.samples.length * 100).toFixed(2);
        
        row.innerHTML = `
            <td>${label}</td>
            <td>${count}</td>
            <td>${percentage}%</td>
        `;
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    
    // Add to container
    container.appendChild(statsDiv);
    container.appendChild(table);
}

/**
 * Show prediction confidence data in tabular format
 */
function showPredictionConfidenceData(data, container) {
    if (!data.predictions || data.predictions.length === 0) {
        container.innerHTML = '<p>No prediction data available</p>';
        return;
    }
    
    // Create statistics info
    const statsDiv = document.createElement('div');
    statsDiv.innerHTML = `
        <p><strong>Total predictions:</strong> ${data.predictions.length}</p>
        <p><strong>Dataset:</strong> ${data.dataset}</p>
        <p><strong>Filter:</strong> ${data.digit ? `Digit ${data.digit}` : 'All digits'}</p>
    `;
    
    // Calculate statistics per digit
    const statsByDigit = {};
    data.predictions.forEach(pred => {
        const trueLabel = pred.true_label;
        const predLabel = pred.predicted_label;
        const confidence = pred.confidence;
        
        if (!statsByDigit[trueLabel]) {
            statsByDigit[trueLabel] = {
                correct: 0,
                total: 0,
                confidences: []
            };
        }
        
        statsByDigit[trueLabel].total++;
        if (trueLabel === predLabel) {
            statsByDigit[trueLabel].correct++;
            statsByDigit[trueLabel].confidences.push(confidence);
        }
    });
    
    // Calculate average confidence per digit
    Object.keys(statsByDigit).forEach(digit => {
        const stats = statsByDigit[digit];
        const confidences = stats.confidences;
        
        stats.accuracy = stats.correct / stats.total;
        stats.avgConfidence = confidences.length > 0 
            ? confidences.reduce((a, b) => a + b, 0) / confidences.length 
            : 0;
    });
    
    // Create statistics table
    const table = document.createElement('table');
    table.className = 'data-table';
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    const headerRowContent = '<th>Digit</th><th>Total</th><th>Correct</th><th>Accuracy</th><th>Avg. Confidence</th>';
    headerRow.innerHTML = headerRowContent;
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    Object.entries(statsByDigit).sort((a, b) => a[0] - b[0]).forEach(([digit, stats]) => {
        const row = document.createElement('tr');
        
        row.innerHTML = `
            <td>${digit}</td>
            <td>${stats.total}</td>
            <td>${stats.correct}</td>
            <td>${(stats.accuracy * 100).toFixed(2)}%</td>
            <td>${(stats.avgConfidence * 100).toFixed(2)}%</td>
        `;
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    
    // Add to container
    container.appendChild(statsDiv);
    container.appendChild(table);
}

/**
 * Show t-SNE data in tabular format
 */
function showTSNEData(data, container) {
    if (!data.tsne_results || data.tsne_results.length === 0) {
        container.innerHTML = '<p>No t-SNE data available</p>';
        return;
    }
    
    // Create statistics info
    const statsDiv = document.createElement('div');
    statsDiv.innerHTML = `
        <p><strong>Total points:</strong> ${data.tsne_results.length}</p>
        <p><strong>Dataset:</strong> ${data.dataset}</p>
        <p><strong>Sample size:</strong> ${data.sample_size}</p>
        <p><strong>Model:</strong> ${data.model_id}</p>
    `;
    
    // Count points per label
    const countByLabel = {};
    data.tsne_results.forEach(point => {
        const label = point.label;
        countByLabel[label] = (countByLabel[label] || 0) + 1;
    });
    
    // Create count by digit table
    const table = document.createElement('table');
    table.className = 'data-table';
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = '<th>Digit</th><th>Count</th><th>Percentage</th>';
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    Object.entries(countByLabel).sort((a, b) => a[0] - b[0]).forEach(([label, count]) => {
        const row = document.createElement('tr');
        const percentage = (count / data.tsne_results.length * 100).toFixed(2);
        
        row.innerHTML = `
            <td>${label}</td>
            <td>${count}</td>
            <td>${percentage}%</td>
        `;
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    
    // Add to container
    container.appendChild(statsDiv);
    container.appendChild(table);
}

/**
 * Show misclassified data in tabular format
 */
function showMisclassifiedData(data, container) {
    if (!data.misclassified || data.misclassified.length === 0) {
        container.innerHTML = '<p>No misclassified examples available</p>';
        return;
    }
    
    // Create statistics info
    const statsDiv = document.createElement('div');
    statsDiv.innerHTML = `
        <p><strong>Total misclassified:</strong> ${data.total_misclassified} out of ${data.total_samples} (${(data.error_rate * 100).toFixed(2)}% error rate)</p>
        <p><strong>Dataset:</strong> ${data.dataset}</p>
        <p><strong>Filter:</strong> ${data.digit ? `Digit ${data.digit}` : 'All digits'}</p>
        <p><strong>Showing:</strong> ${data.misclassified.length} examples</p>
    `;
    
    // Count misclassifications by true label and predicted label
    const confusionCount = {};
    data.misclassified.forEach(example => {
        const trueLabel = example.true_label;
        const predLabel = example.predicted_label;
        
        if (!confusionCount[trueLabel]) {
            confusionCount[trueLabel] = {};
        }
        
        if (!confusionCount[trueLabel][predLabel]) {
            confusionCount[trueLabel][predLabel] = 0;
        }
        
        confusionCount[trueLabel][predLabel]++;
    });
    
    // Create confusion table
    const table = document.createElement('table');
    table.className = 'data-table';
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = '<th>True \ Predicted</th>';
    
    // Add predicted label headers (0-9)
    for (let i = 0; i < 10; i++) {
        const th = document.createElement('th');
        th.textContent = i;
        headerRow.appendChild(th);
    }
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    
    // For each true label (0-9)
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('tr');
        
        // Add row header (true label)
        const rowHeader = document.createElement('th');
        rowHeader.textContent = i;
        row.appendChild(rowHeader);
        
        // Add cells for each predicted label
        for (let j = 0; j < 10; j++) {
            const td = document.createElement('td');
            
            // Skip diagonal (correct classifications)
            if (i === j) {
                td.textContent = '-';
                td.style.backgroundColor = '#f8f8f8';
            } else {
                const count = confusionCount[i] && confusionCount[i][j] ? confusionCount[i][j] : 0;
                td.textContent = count;
                
                // Highlight high counts
                if (count > 0) {
                    const intensity = Math.min(1, count / 10);
                    td.style.backgroundColor = `rgba(74, 108, 247, ${intensity * 0.2})`;
                    td.style.fontWeight = intensity > 0.5 ? 'bold' : 'normal';
                }
            }
            
            row.appendChild(td);
        }
        
        tbody.appendChild(row);
    }
    
    table.appendChild(tbody);
    
    // Add to container
    container.appendChild(statsDiv);
    container.appendChild(table);
}

/**
 * Generate insights based on visualization type
 */
function generateInsights(data) {
    const container = document.getElementById('insights-list');
    container.innerHTML = '';
    
    if (!data) {
        container.innerHTML = '<p>No data available to generate insights</p>';
        return;
    }
    
    switch (state.visualizationType) {
        case 'confusion-matrix':
            generateConfusionMatrixInsights(data, container);
            break;
        case 'digit-samples':
            generateDigitSamplesInsights(data, container);
            break;
        case 'prediction-confidence':
            generatePredictionConfidenceInsights(data, container);
            break;
        case 'tsne':
            generateTSNEInsights(data, container);
            break;
        case 'misclassified':
            generateMisclassifiedInsights(data, container);
            break;
    }
}

/**
 * Generate insights for confusion matrix
 */
function generateConfusionMatrixInsights(data, container) {
    if (!data.confusion_matrix) return;
    
    const matrix = data.confusion_matrix;
    const insights = [];
    
    // Calculate per-digit accuracy
    const digitAccuracy = [];
    let maxAccuracy = 0;
    let minAccuracy = 1;
    let maxAccuracyDigit = -1;
    let minAccuracyDigit = -1;
    
    for (let i = 0; i < 10; i++) {
        let total = 0;
        const correct = matrix[i][i];
        
        for (let j = 0; j < 10; j++) {
            total += matrix[i][j];
        }
        
        const accuracy = total > 0 ? correct / total : 0;
        digitAccuracy.push(accuracy);
        
        if (accuracy > maxAccuracy) {
            maxAccuracy = accuracy;
            maxAccuracyDigit = i;
        }
        
        if (accuracy < minAccuracy) {
            minAccuracy = accuracy;
            minAccuracyDigit = i;
        }
    }
    
    // Find most common misclassifications
    const misclassifications = [];
    
    for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 10; j++) {
            if (i !== j && matrix[i][j] > 0) {
                misclassifications.push({
                    from: i,
                    to: j,
                    count: matrix[i][j],
                    total: matrix[i].reduce((a, b) => a + b, 0)
                });
            }
        }
    }
    
    // Sort by count (descending)
    misclassifications.sort((a, b) => b.count - a.count);
    
    // Add overall accuracy insight
    insights.push({
        title: 'Overall Model Performance',
        text: `The model achieves an overall accuracy of ${(data.accuracy * 100).toFixed(2)}% on the ${state.dataset} dataset.`
    });
    
    // Add best/worst performing digits
    insights.push({
        title: 'Digit Recognition Performance',
        text: `
            <ul>
                <li>Best recognized digit: ${maxAccuracyDigit} (${(maxAccuracy * 100).toFixed(2)}% accuracy)</li>
                <li>Most challenging digit: ${minAccuracyDigit} (${(minAccuracy * 100).toFixed(2)}% accuracy)</li>
            </ul>
        `
    });
    
    // Add common misclassifications insight
    if (misclassifications.length > 0) {
        const topMisclassifications = misclassifications.slice(0, 3);
        
        insights.push({
            title: 'Common Misclassifications',
            text: `
                <ul>
                    ${topMisclassifications.map(m => 
                        `<li>Digit ${m.from} is mistaken for ${m.to} in ${m.count} cases (${((m.count / m.total) * 100).toFixed(1)}% of all ${m.from}s)</li>`
                    ).join('')}
                </ul>
            `
        });
    }
    
    // Add insights to container
    insights.forEach(insight => {
        const insightDiv = document.createElement('div');
        insightDiv.className = 'insight-item';
        insightDiv.innerHTML = `
            <h4>${insight.title}</h4>
            <div>${insight.text}</div>
        `;
        container.appendChild(insightDiv);
    });
}

/**
 * Generate insights for digit samples
 */
function generateDigitSamplesInsights(data, container) {
    if (!data.samples || data.samples.length === 0) {
        return;
    }
    
    const samples = data.samples;
    const insights = [];
    
    // Count samples per digit
    const digitCounts = {};
    samples.forEach(sample => {
        const digit = sample.true_label;
        digitCounts[digit] = (digitCounts[digit] || 0) + 1;
    });
    
    // Add dataset information insight
    insights.push({
        title: 'Dataset Information',
        text: `These samples are from the ${data.dataset} dataset, which contains ${data.total_count.toLocaleString()} images.
              ${state.digit !== 'all' ? `Filtered to show only digit ${state.digit}, which has ${data.filtered_count.toLocaleString()} examples in this dataset.` : ''}`
    });
    
    // Add sample distribution insight
    const distributionText = Object.entries(digitCounts)
        .sort((a, b) => a[0] - b[0])
        .map(([digit, count]) => 
            `<li>Digit <strong>${digit}</strong>: ${count} samples (${(count / samples.length * 100).toFixed(2)}%)</li>`
        )
        .join('');
    
    insights.push({
        title: 'Sample Distribution',
        text: `
            <p>Distribution of digits in the current sample:</p>
            <ul>${distributionText}</ul>
        `
    });
    
    // Add usage tips insight
    insights.push({
        title: 'Usage Tips',
        text: 'Click on individual digits to see them in more detail. You can export this visualization as data using the export button below.'
    });
    
    // Add insights to container
    insights.forEach(insight => {
        const insightDiv = document.createElement('div');
        insightDiv.className = 'insight-item';
        insightDiv.innerHTML = `
            <h4>${insight.title}</h4>
            <div>${insight.text}</div>
        `;
        container.appendChild(insightDiv);
    });
}

/**
 * Generate insights for prediction confidence
 */
function generatePredictionConfidenceInsights(data, container) {
    if (!data.predictions || data.predictions.length === 0) {
        return;
    }
    
    const predictions = data.predictions;
    const insights = [];
    
    // Calculate statistics per digit
    const statsByDigit = {};
    predictions.forEach(pred => {
        const trueLabel = pred.true_label;
        const predLabel = pred.predicted_label;
        const confidence = pred.confidence;
        
        if (!statsByDigit[trueLabel]) {
            statsByDigit[trueLabel] = {
                correct: 0,
                total: 0,
                confidences: []
            };
        }
        
        statsByDigit[trueLabel].total++;
        if (trueLabel === predLabel) {
            statsByDigit[trueLabel].correct++;
            statsByDigit[trueLabel].confidences.push(confidence);
        }
    });
    
    // Calculate average confidence and identify high/low confidence digits
    let highestAvgConf = 0;
    let lowestAvgConf = 1;
    let highestDigit = -1;
    let lowestDigit = -1;
    
    Object.entries(statsByDigit).forEach(([digit, stats]) => {
        const confidences = stats.confidences;
        if (confidences.length > 0) {
            const avgConf = confidences.reduce((a, b) => a + b, 0) / confidences.length;
            stats.avgConfidence = avgConf;
            
            if (avgConf > highestAvgConf) {
                highestAvgConf = avgConf;
                highestDigit = digit;
            }
            
            if (avgConf < lowestAvgConf) {
                lowestAvgConf = avgConf;
                lowestDigit = digit;
            }
        }
    });
    
    // Add overall confidence insight
    const correctPredictions = predictions.filter(p => p.true_label === p.predicted_label);
    const avgOverallConf = correctPredictions.length > 0
        ? correctPredictions.reduce((sum, p) => sum + p.confidence, 0) / correctPredictions.length
        : 0;
    
    insights.push({
        title: 'Overall Confidence',
        text: `The model predicts correct digits with an average confidence of ${(avgOverallConf * 100).toFixed(2)}%.`
    });
    
    // Add high/low confidence insights
    if (highestDigit !== -1 && lowestDigit !== -1) {
        insights.push({
            title: 'Confidence by Digit',
            text: `
                <ul>
                    <li>The model is most confident when predicting the digit <strong>${highestDigit}</strong> 
                        (average confidence: ${(highestAvgConf * 100).toFixed(2)}%).</li>
                    <li>The model is least confident when predicting the digit <strong>${lowestDigit}</strong> 
                        (average confidence: ${(lowestAvgConf * 100).toFixed(2)}%).</li>
                </ul>
            `
        });
    }
    
    // Find cases of high confidence but incorrect predictions
    const highConfIncorrect = predictions.filter(p => 
        p.true_label !== p.predicted_label && p.confidence > 0.9
    );
    
    if (highConfIncorrect.length > 0) {
        insights.push({
            title: 'Potential Overconfidence',
            text: `The model shows high confidence (>90%) in ${highConfIncorrect.length} incorrect predictions. 
                  This may indicate areas where the model is overconfident in its mistakes.`
        });
    }
    
    // Add insights to container
    insights.forEach(insight => {
        const insightDiv = document.createElement('div');
        insightDiv.className = 'insight-item';
        insightDiv.innerHTML = `
            <h4>${insight.title}</h4>
            <div>${insight.text}</div>
        `;
        container.appendChild(insightDiv);
    });
}

/**
 * Generate insights for t-SNE
 */
function generateTSNEInsights(data, container) {
    if (!data.tsne_results || data.tsne_results.length === 0) {
        return;
    }
    
    const insights = [];
    
    // Add general t-SNE insights
    insights.push({
        title: 'Understanding t-SNE Visualization',
        text: `
            <p>This t-SNE plot reveals how the model organizes the MNIST digits in a 2D space:</p>
            <ul>
                <li>Clusters of the same color represent digits that the model perceives as similar.</li>
                <li>The distance between points approximates their similarity in the model's feature space.</li>
                <li>Isolated points may represent atypical or difficult examples of a digit.</li>
            </ul>
        `
    });
    
    // Count points per label and identify clusters
    const countByLabel = {};
    data.tsne_results.forEach(point => {
        const label = point.label;
        countByLabel[label] = (countByLabel[label] || 0) + 1;
    });
    
    insights.push({
        title: 'Visualization Details',
        text: `
            <p>This visualization is based on ${data.sample_size} randomly selected examples from the ${data.dataset} dataset.</p>
            <p>The features used for t-SNE were ${data.model_id === 'raw_pixels' ? 'raw pixel values' : 'extracted from the hidden layer of the model'}.</p>
        `
    });
    
    // Add interpretation tips
    insights.push({
        title: 'Interpretation Tips',
        text: `
            <ul>
                <li>Well-separated clusters suggest the model can easily distinguish between those digits.</li>
                <li>Overlapping clusters indicate digits that the model may confuse.</li>
                <li>The overall structure reflects how the model "perceives" digit similarities.</li>
            </ul>
        `
    });
    
    // Add insights to container
    insights.forEach(insight => {
        const insightDiv = document.createElement('div');
        insightDiv.className = 'insight-item';
        insightDiv.innerHTML = `
            <h4>${insight.title}</h4>
            <div>${insight.text}</div>
        `;
        container.appendChild(insightDiv);
    });
}

/**
 * Generate insights for misclassified examples
 */
function generateMisclassifiedInsights(data, container) {
    if (!data.misclassified || data.misclassified.length === 0) {
        return;
    }
    
    const misclassified = data.misclassified;
    const insights = [];
    
    // Count misclassifications by true label and predicted label
    const confusionCount = {};
    const totalByTrue = {};
    
    misclassified.forEach(example => {
        const trueLabel = example.true_label;
        const predLabel = example.predicted_label;
        
        if (!confusionCount[trueLabel]) {
            confusionCount[trueLabel] = {};
        }
        
        if (!confusionCount[trueLabel][predLabel]) {
            confusionCount[trueLabel][predLabel] = 0;
        }
        
        confusionCount[trueLabel][predLabel]++;
        totalByTrue[trueLabel] = (totalByTrue[trueLabel] || 0) + 1;
    });
    
    // Find most common misclassifications
    const confusions = [];
    
    Object.entries(confusionCount).forEach(([trueLabel, predictions]) => {
        Object.entries(predictions).forEach(([predLabel, count]) => {
            confusions.push({
                from: parseInt(trueLabel),
                to: parseInt(predLabel),
                count: count,
                percentage: count / totalByTrue[trueLabel]
            });
        });
    });
    
    // Sort by count (descending)
    confusions.sort((a, b) => b.count - a.count);
    
    // Add error rate insight
    insights.push({
        title: 'Model Error Analysis',
        text: `The model has an overall error rate of ${(data.error_rate * 100).toFixed(2)}% on the ${data.dataset} dataset.`
    });
    
    // Add common confusions insight
    if (confusions.length > 0) {
        const topConfusions = confusions.slice(0, 5);
        
        insights.push({
            title: 'Common Misclassifications',
            text: `
                <p>The most common misclassifications are:</p>
                <ul>
                    ${topConfusions.map(c => 
                        `<li>Digit <strong>${c.from}</strong> mistaken as <strong>${c.to}</strong>: 
                        ${c.count} examples (${(c.percentage * 100).toFixed(1)}% of misclassified ${c.from}s)</li>`
                    ).join('')}
                </ul>
            `
        });
    }
    
    // Add confidence insight
    const avgConfidence = misclassified.reduce((sum, ex) => sum + ex.confidence, 0) / misclassified.length;
    
    insights.push({
        title: 'Confidence in Errors',
        text: `The model's average confidence in its incorrect predictions is ${(avgConfidence * 100).toFixed(2)}%.
              ${avgConfidence > 0.8 ? 'This high confidence in errors suggests the model may be overconfident in some cases.' : 
                'This moderate/low confidence suggests the model has some uncertainty about these predictions.'}`
    });
    
    // Add usage tip
    insights.push({
        title: 'Usage Tip',
        text: 'Click on any misclassified example to see details, including the model\'s confidence scores for all digit classes.'
    });
    
    // Add insights to container
    insights.forEach(insight => {
        const insightDiv = document.createElement('div');
        insightDiv.className = 'insight-item';
        insightDiv.innerHTML = `
            <h4>${insight.title}</h4>
            <div>${insight.text}</div>
        `;
        container.appendChild(insightDiv);
    });
}

/**
 * Export visualization as image
 */
function exportAsImage() {
    const visualizationContent = document.getElementById('visualization-content');
    
    // Check if Plotly visualization
    if (visualizationContent.data && visualizationContent.data[0] && visualizationContent.data[0].layout) {
        Plotly.downloadImage(visualizationContent, {
            format: 'png',
            filename: `mnist_${state.visualizationType}_${new Date().getTime()}`
        });
    } else {
        // For non-Plotly visualizations, use html2canvas
        alert("Export as image is only available for chart visualizations");
    }
}

/**
 * Export visualization data
 */
function exportData() {
    if (!state.currentData) {
        showError('No data available to export');
        return;
    }
    
    // Create a JSON string
    const dataStr = JSON.stringify(state.currentData, null, 2);
    
    // Create a blob and download link
    const blob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `mnist_${state.visualizationType}_${new Date().getTime()}.json`;
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
}

/**
 * Show/hide loading spinner
 */
function showLoading(isLoading) {
    document.getElementById('visualization-loading').style.display = isLoading ? 'block' : 'none';
}

/**
 * Show error message
 */
function showError(message) {
    const container = document.getElementById('visualization-content');
    container.innerHTML = `
        <div class="error-message">
            <p>${message}</p>
        </div>
    `;
}

/**
 * Generate model weights visualization
 */
function generateModelWeights() {
    // Prepare request data
    const requestData = {
        model_id: state.currentModel === 'current' ? null : state.currentModel
    };
    
    // Make API request
    fetch('/api/visualize/model_weights', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            state.currentData = data;
            renderModelWeights(data);
            
            // Show insights
            document.querySelector('.insights-section').style.display = 'block';
            generateWeightInsights(data);
            
            // Show data
            document.getElementById('data-section').style.display = 'block';
            showDataTab(data);
        } else {
            showError('Failed to generate model weights visualization: ' + data.message);
        }
        showLoading(false);
    })
    .catch(error => {
        console.error('Error generating model weights:', error);
        showError('Failed to generate model weights visualization. Check console for details.');
        showLoading(false);
    });
}

/**
 * Render model weights visualization
 */
function renderModelWeights(data) {
    // 清空可视化容器
    const container = document.getElementById('visualization-content');
    container.innerHTML = '';
    
    // 显示标题和容器
    document.getElementById('visualization-placeholder').style.display = 'none';
    document.getElementById('visualization-container').style.display = 'block';
    document.getElementById('visualization-title').textContent = 'Neural Network Weights & Architecture';
    
    // 添加模型架构信息
    const architectureDiv = document.createElement('div');
    architectureDiv.className = 'architecture-info';
    architectureDiv.innerHTML = `
        <h3>Model Architecture</h3>
        <div class="info-grid">
            <div class="info-item">
                <span class="info-label">Input Size:</span>
                <span class="info-value">${data.architecture.input_size} (28×28 pixels)</span>
            </div>
            <div class="info-item">
                <span class="info-label">Hidden Layer Size:</span>
                <span class="info-value">${data.architecture.hidden_size} neurons</span>
            </div>
            <div class="info-item">
                <span class="info-label">Output Size:</span>
                <span class="info-value">${data.architecture.output_size} (10 digits)</span>
            </div>
            <div class="info-item">
                <span class="info-label">Activation Function:</span>
                <span class="info-value">${data.architecture.activation}</span>
            </div>
        </div>
    `;
    container.appendChild(architectureDiv);
    
    // 创建网络架构可视化
    const networkDiv = document.createElement('div');
    networkDiv.className = 'network-visualization';
    networkDiv.innerHTML = `
        <h3>Network Architecture Visualization</h3>
        <div id="network-viz"></div>
    `;
    container.appendChild(networkDiv);
    
    // 创建权重可视化区域
    const weightsDiv = document.createElement('div');
    weightsDiv.className = 'weights-visualization';
    weightsDiv.innerHTML = `
        <h3>Layer Weights</h3>
        <div class="layer-tabs">
            <button id="layer1-tab" class="layer-tab active">Input → Hidden Layer</button>
            <button id="layer2-tab" class="layer-tab">Hidden → Output Layer</button>
        </div>
        <div id="weight-viz"></div>
    `;
    container.appendChild(weightsDiv);
    
    // 创建权重统计信息区域
    const statsDiv = document.createElement('div');
    statsDiv.className = 'weights-stats';
    statsDiv.innerHTML = `
        <h3>Weight Statistics</h3>
        <div class="stats-container">
            <div class="stats-card">
                <h4>Layer 1 (Input → Hidden)</h4>
                <div class="stats-grid">
                    <div class="stats-item">
                        <span class="stats-label">Min:</span>
                        <span class="stats-value">${data.stats.layer1.min.toFixed(4)}</span>
                    </div>
                    <div class="stats-item">
                        <span class="stats-label">Max:</span>
                        <span class="stats-value">${data.stats.layer1.max.toFixed(4)}</span>
                    </div>
                    <div class="stats-item">
                        <span class="stats-label">Mean:</span>
                        <span class="stats-value">${data.stats.layer1.mean.toFixed(4)}</span>
                    </div>
                    <div class="stats-item">
                        <span class="stats-label">Std:</span>
                        <span class="stats-value">${data.stats.layer1.std.toFixed(4)}</span>
                    </div>
                </div>
            </div>
            <div class="stats-card">
                <h4>Layer 2 (Hidden → Output)</h4>
                <div class="stats-grid">
                    <div class="stats-item">
                        <span class="stats-label">Min:</span>
                        <span class="stats-value">${data.stats.layer2.min.toFixed(4)}</span>
                    </div>
                    <div class="stats-item">
                        <span class="stats-label">Max:</span>
                        <span class="stats-value">${data.stats.layer2.max.toFixed(4)}</span>
                    </div>
                    <div class="stats-item">
                        <span class="stats-label">Mean:</span>
                        <span class="stats-value">${data.stats.layer2.mean.toFixed(4)}</span>
                    </div>
                    <div class="stats-item">
                        <span class="stats-label">Std:</span>
                        <span class="stats-value">${data.stats.layer2.std.toFixed(4)}</span>
                    </div>
                </div>
            </div>
        </div>
    `;
    container.appendChild(statsDiv);
    
    // 绘制网络架构可视化
    drawNetworkArchitecture(data.architecture);
    
    // 默认显示第一层权重
    drawWeightHeatmap('layer1', data.weights.layer1);
    
    // 添加层切换事件
    document.getElementById('layer1-tab').addEventListener('click', () => {
        document.querySelectorAll('.layer-tab').forEach(tab => tab.classList.remove('active'));
        document.getElementById('layer1-tab').classList.add('active');
        drawWeightHeatmap('layer1', data.weights.layer1);
    });
    
    document.getElementById('layer2-tab').addEventListener('click', () => {
        document.querySelectorAll('.layer-tab').forEach(tab => tab.classList.remove('active'));
        document.getElementById('layer2-tab').classList.add('active');
        drawWeightHeatmap('layer2', data.weights.layer2);
    });
    
    // 添加可视化描述
    document.getElementById('visualization-caption').innerHTML = `
        <p>This visualization shows the architecture and learned weights of the neural network model. 
        The network consists of an input layer (${data.architecture.input_size} neurons representing the flattened 28×28 MNIST images), 
        a hidden layer with ${data.architecture.hidden_size} neurons using ${data.architecture.activation} activation, 
        and an output layer with ${data.architecture.output_size} neurons (one for each digit 0-9).</p>
        <p>The heatmaps show the weight matrices that connect layers. Brighter colors indicate larger positive weights, 
        darker colors indicate larger negative weights.</p>
    `;
}

/**
 * Draw neural network architecture visualization
 */
function drawNetworkArchitecture(architecture) {
    const container = document.getElementById('network-viz');
    
    // 设置SVG容器
    const width = container.clientWidth;
    const height = 300;
    
    // 创建SVG
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .append('g')
        .attr('transform', `translate(0, 10)`);
    
    // 定义层节点数
    const layers = [
        {name: 'Input Layer', size: architecture.input_size},
        {name: 'Hidden Layer', size: architecture.hidden_size},
        {name: 'Output Layer', size: architecture.output_size}
    ];
    
    // 计算每层可以显示的最大节点数
    const maxNodesToShow = {
        'Input Layer': Math.min(10, architecture.input_size),
        'Hidden Layer': Math.min(10, architecture.hidden_size),
        'Output Layer': architecture.output_size
    };
    
    // 设置水平间距
    const layerSpacing = width / (layers.length + 1);
    
    // 绘制连接线（示意性的，不是实际权重）
    for (let i = 0; i < layers.length - 1; i++) {
        const sourceNodes = Math.min(maxNodesToShow[layers[i].name], layers[i].size);
        const targetNodes = Math.min(maxNodesToShow[layers[i+1].name], layers[i+1].size);
        
        // 计算起始和结束位置
        const sourceX = (i + 1) * layerSpacing;
        const targetX = (i + 2) * layerSpacing;
        
        // 确定每层节点的垂直间距
        const sourceSpacing = height / (sourceNodes + 1);
        const targetSpacing = height / (targetNodes + 1);
        
        // 连接源节点和目标节点
        for (let s = 0; s < sourceNodes; s++) {
            // 如果节点过多，只显示一部分
            if (s > 8 && s < layers[i].size - 1) {
                if (s === 9) {
                    // 添加省略号
                    svg.append('text')
                        .attr('x', sourceX)
                        .attr('y', (s + 1) * sourceSpacing + 5)
                        .attr('text-anchor', 'middle')
                        .attr('font-size', '14px')
                        .text('...');
                }
                continue;
            }
            
            const sourceY = (s + 1) * sourceSpacing;
            
            for (let t = 0; t < targetNodes; t++) {
                // 如果节点过多，只显示一部分
                if (t > 8 && t < layers[i+1].size - 1) continue;
                
                const targetY = (t + 1) * targetSpacing;
                
                // 绘制连接线
                svg.append('line')
                    .attr('x1', sourceX)
                    .attr('y1', sourceY)
                    .attr('x2', targetX)
                    .attr('y2', targetY)
                    .attr('stroke', '#ccc')
                    .attr('stroke-width', 0.5)
                    .attr('opacity', 0.5);
            }
        }
    }
    
    // 绘制每一层的节点
    for (let i = 0; i < layers.length; i++) {
        const x = (i + 1) * layerSpacing;
        const layerSize = layers[i].size;
        const nodesToShow = Math.min(maxNodesToShow[layers[i].name], layerSize);
        const spacing = height / (nodesToShow + 1);
        
        // 添加层标签
        svg.append('text')
            .attr('x', x)
            .attr('y', 20)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .text(`${layers[i].name} (${layerSize})`);
        
        // 添加节点
        for (let j = 0; j < nodesToShow; j++) {
            // 如果节点过多，只显示一部分
            if (j > 8 && j < layerSize - 1) continue;
            
            const y = (j + 1) * spacing;
            
            // 绘制圆圈节点
            svg.append('circle')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', 8)
                .attr('fill', getLayerColor(i))
                .attr('stroke', '#666')
                .attr('stroke-width', 1);
        }
    }
}

/**
 * Get color for different network layers
 */
function getLayerColor(layerIndex) {
    const colors = ['#4a6cf7', '#f7941d', '#3bceac'];
    return colors[layerIndex % colors.length];
}

/**
 * Draw weight matrix heatmap
 */
function drawWeightHeatmap(layerId, layerData) {
    const container = document.getElementById('weight-viz');
    container.innerHTML = '';
    
    // 获取权重矩阵
    const weights = layerData.weights;
    const shape = layerData.shape;
    const bias = layerData.bias;
    
    // 如果权重矩阵太大，进行下采样
    let sampledWeights = weights;
    let sampleFactor = 1;
    let title = '';
    
    if (layerId === 'layer1') {
        title = 'Input → Hidden Layer Weights';
        // 输入层权重可以作为28x28像素的过滤器可视化
        if (shape[0] === 784) {  // 28*28
            const weightsDiv = document.createElement('div');
            weightsDiv.className = 'neuron-filters';
            weightsDiv.innerHTML = `<h4>Hidden Layer Neuron Filters (Sampling)</h4>`;
            container.appendChild(weightsDiv);
            
            // 只显示一部分神经元的过滤器（最多25个）
            const numNeuronsToShow = Math.min(shape[1], 25);
            const neuronsToShow = [];
            
            // 选择均匀分布的神经元
            if (shape[1] <= 25) {
                for (let i = 0; i < shape[1]; i++) {
                    neuronsToShow.push(i);
                }
            } else {
                const step = shape[1] / 25;
                for (let i = 0; i < 25; i++) {
                    neuronsToShow.push(Math.floor(i * step));
                }
            }
            
            // 创建过滤器网格
            const filterGrid = document.createElement('div');
            filterGrid.className = 'filter-grid';
            weightsDiv.appendChild(filterGrid);
            
            // 为每个选择的神经元绘制过滤器
            for (let i = 0; i < neuronsToShow.length; i++) {
                const neuronIdx = neuronsToShow[i];
                const neuronWeights = [];
                
                // 将1D权重重构为28x28的2D过滤器
                for (let pixel = 0; pixel < 784; pixel++) {
                    neuronWeights.push(weights[pixel][neuronIdx]);
                }
                
                // 创建过滤器可视化
                const filterContainer = document.createElement('div');
                filterContainer.className = 'filter-container';
                filterContainer.innerHTML = `<div class="filter-label">Neuron ${neuronIdx}</div>`;
                filterGrid.appendChild(filterContainer);
                
                // 绘制过滤器
                drawNeuronFilter(filterContainer, neuronWeights, 28, 28);
            }
            
            // 添加下采样的权重矩阵热力图
            // 创建一个小型的权重矩阵热力图
            const smallHeatmapDiv = document.createElement('div');
            smallHeatmapDiv.className = 'weight-matrix-heatmap';
            smallHeatmapDiv.innerHTML = `<h4>Weight Matrix (${shape[0]}×${shape[1]}) - Downsampled View</h4>`;
            container.appendChild(smallHeatmapDiv);
            
            // 采样权重矩阵
            const maxDim = 100;  // 最大可视化尺寸
            sampleFactor = Math.ceil(Math.max(shape[0], shape[1]) / maxDim);
            
            // 创建下采样的权重矩阵
            const downsampledWeights = downsampleMatrix(weights, sampleFactor);
            
            // 绘制热力图
            drawHeatmap(smallHeatmapDiv, downsampledWeights);
            
            // 添加说明
            const explanationDiv = document.createElement('div');
            explanationDiv.className = 'weight-explanation';
            explanationDiv.innerHTML = `
                <p>The filters above show how each hidden neuron processes the input. 
                   Bright areas indicate where the neuron looks for positive features, 
                   dark areas indicate negative features.</p>
                <p>The heatmap shows a downsampled view of the full weight matrix 
                   (sampled by factor ${sampleFactor}).</p>
                <p>This layer has ${shape[1]} bias values, ranging from 
                   ${Math.min(...bias).toFixed(4)} to ${Math.max(...bias).toFixed(4)}.</p>
            `;
            container.appendChild(explanationDiv);
            
            return;
        }
    } else if (layerId === 'layer2') {
        title = 'Hidden → Output Layer Weights';
    }
    
    // 普通矩阵热力图
    const heatmapDiv = document.createElement('div');
    heatmapDiv.className = 'weight-matrix';
    heatmapDiv.innerHTML = `<h4>${title} (${shape[0]}×${shape[1]})</h4>`;
    container.appendChild(heatmapDiv);
    
    // 检查矩阵大小，如果太大则进行下采样
    const maxDim = 100;  // 最大可视化尺寸
    if (shape[0] > maxDim || shape[1] > maxDim) {
        sampleFactor = Math.ceil(Math.max(shape[0], shape[1]) / maxDim);
        sampledWeights = downsampleMatrix(weights, sampleFactor);
        
        const noticeDiv = document.createElement('div');
        noticeDiv.className = 'sampling-notice';
        noticeDiv.innerHTML = `<p>The weight matrix is downsampled by factor ${sampleFactor} for visualization.</p>`;
        heatmapDiv.appendChild(noticeDiv);
    }
    
    // 绘制热力图
    drawHeatmap(heatmapDiv, sampledWeights);
    
    // 添加偏置可视化
    const biasDiv = document.createElement('div');
    biasDiv.className = 'bias-visualization';
    biasDiv.innerHTML = `
        <h4>Bias Values</h4>
        <div id="bias-viz"></div>
    `;
    container.appendChild(biasDiv);
    
    // 绘制偏置条形图
    drawBiasBarChart(document.getElementById('bias-viz'), bias);
    
    // 添加说明
    const explanationDiv = document.createElement('div');
    explanationDiv.className = 'weight-explanation';
    explanationDiv.innerHTML = `
        <p>This heatmap shows the weight matrix connecting ${layerId === 'layer1' ? 'input to hidden' : 'hidden to output'} layer. 
           Each column represents the weights for one ${layerId === 'layer1' ? 'hidden' : 'output'} neuron.</p>
        <p>Bright colors indicate positive weights, dark colors indicate negative weights.</p>
        <p>This layer has ${bias.length} bias values, ranging from 
           ${Math.min(...bias).toFixed(4)} to ${Math.max(...bias).toFixed(4)}.</p>
    `;
    container.appendChild(explanationDiv);
}

/**
 * Downsample a matrix for visualization
 */
function downsampleMatrix(matrix, factor) {
    if (factor <= 1) return matrix;
    
    const rows = matrix.length;
    const cols = matrix[0].length;
    
    const newRows = Math.ceil(rows / factor);
    const newCols = Math.ceil(cols / factor);
    
    const result = Array(newRows).fill().map(() => Array(newCols).fill(0));
    
    for (let i = 0; i < newRows; i++) {
        for (let j = 0; j < newCols; j++) {
            // 计算采样窗口的范围
            const startRow = i * factor;
            const endRow = Math.min((i + 1) * factor, rows);
            const startCol = j * factor;
            const endCol = Math.min((j + 1) * factor, cols);
            
            // 计算采样窗口内的平均值
            let sum = 0;
            let count = 0;
            
            for (let r = startRow; r < endRow; r++) {
                for (let c = startCol; c < endCol; c++) {
                    sum += matrix[r][c];
                    count++;
                }
            }
            
            result[i][j] = sum / count;
        }
    }
    
    return result;
}

/**
 * Draw a heatmap visualization using Plotly
 */
function drawHeatmap(container, matrix) {
    // 获取值的范围以设置对称的色标范围
    let maxAbs = 0;
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
            maxAbs = Math.max(maxAbs, Math.abs(matrix[i][j]));
        }
    }
    
    // 创建热力图数据
    const data = [{
        z: matrix,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmin: -maxAbs,
        zmax: maxAbs
    }];
    
    const layout = {
        margin: {t: 30, r: 20, b: 30, l: 30},
        height: 400,
        xaxis: {title: 'Neuron Index', showticklabels: matrix[0].length < 50},
        yaxis: {title: 'Input Index', showticklabels: matrix.length < 50}
    };
    
    Plotly.newPlot(container, data, layout);
}

/**
 * Draw neuron filter visualization using Canvas
 */
function drawNeuronFilter(container, weights, width, height) {
    // 创建一个Canvas元素
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    canvas.className = 'neuron-filter';
    container.appendChild(canvas);
    
    // 获取Canvas 2D上下文
    const ctx = canvas.getContext('2d');
    
    // 计算权重的范围
    let minWeight = Infinity;
    let maxWeight = -Infinity;
    
    for (let i = 0; i < weights.length; i++) {
        minWeight = Math.min(minWeight, weights[i]);
        maxWeight = Math.max(maxWeight, weights[i]);
    }
    
    // 为了更好的可视化对比，使用对称的范围
    const absMax = Math.max(Math.abs(minWeight), Math.abs(maxWeight));
    
    // 绘制权重
    const imageData = ctx.createImageData(width, height);
    
    for (let i = 0; i < weights.length; i++) {
        const row = Math.floor(i / width);
        const col = i % width;
        
        // 将权重映射到灰度值 [0, 255]
        // 将范围从[-absMax, absMax]映射到[0, 255]
        const normalizedValue = (weights[i] + absMax) / (2 * absMax);
        const value = Math.round(normalizedValue * 255);
        
        // 设置像素值（RGBA）
        const pixelIndex = (row * width + col) * 4;
        imageData.data[pixelIndex] = value;     // R
        imageData.data[pixelIndex + 1] = value; // G
        imageData.data[pixelIndex + 2] = value; // B
        imageData.data[pixelIndex + 3] = 255;   // A (不透明)
    }
    
    ctx.putImageData(imageData, 0, 0);
}

/**
 * Draw bias values as a bar chart
 */
function drawBiasBarChart(container, biasValues) {
    // 创建数据
    const data = [{
        y: biasValues,
        type: 'bar',
        marker: {
            color: biasValues.map(v => v >= 0 ? '#4a6cf7' : '#f7941d')
        }
    }];
    
    const layout = {
        margin: {t: 10, r: 10, b: 40, l: 50},
        height: 200,
        xaxis: {title: 'Neuron Index'},
        yaxis: {title: 'Bias Value'}
    };
    
    Plotly.newPlot(container, data, layout);
}

/**
 * Generate insights for weight visualization
 */
function generateWeightInsights(data) {
    const insightsList = document.getElementById('insights-list');
    insightsList.innerHTML = '';
    
    // 层1的权重分析
    const layer1 = data.weights.layer1;
    const layer1Stats = data.stats.layer1;
    
    // 层2的权重分析
    const layer2 = data.weights.layer2;
    const layer2Stats = data.stats.layer2;
    
    // 神经元角色分析
    const hiddenActivations = [];
    let activationInfo = '';
    
    // 对于输入→隐藏层，可以分析每个隐藏神经元对于输入图像特征的敏感性
    if (layer1.weights.length === 784) {  // 28*28 pixels
        const insights = [
            {
                title: 'Weight Distribution',
                content: `The input-to-hidden layer weights have a mean of ${layer1Stats.mean.toFixed(4)} with standard deviation ${layer1Stats.std.toFixed(4)}.
                          The hidden-to-output layer weights have a mean of ${layer2Stats.mean.toFixed(4)} with standard deviation ${layer2Stats.std.toFixed(4)}.`
            },
            {
                title: 'Weight Patterns',
                content: `The visualization shows how individual hidden neurons respond to different parts of the input image. 
                          Each filter represents what features a hidden neuron is looking for in the input image.`
            },
            {
                title: 'Network Structure',
                content: `This is a simple 3-layer neural network with ${data.architecture.input_size} input neurons, 
                          ${data.architecture.hidden_size} hidden neurons using ${data.architecture.activation} activation,
                          and ${data.architecture.output_size} output neurons using softmax activation for digit classification.`
            }
        ];
        
        // 添加特定于激活函数的见解
        if (data.architecture.activation === 'relu') {
            insights.push({
                title: 'ReLU Activation',
                content: `The ReLU activation function used in the hidden layer allows the network to learn non-linear patterns.
                          ReLU neurons only activate when their input is positive, creating sparse activations.`
            });
        } else if (data.architecture.activation === 'sigmoid') {
            insights.push({
                title: 'Sigmoid Activation',
                content: `The Sigmoid activation function used in the hidden layer squashes values between 0 and 1.
                          This can help regulate activations but may suffer from vanishing gradient problems during training.`
            });
        }
        
        // 添加所有洞察
        insights.forEach(insight => {
            const insightItem = document.createElement('div');
            insightItem.className = 'insight-item';
            insightItem.innerHTML = `
                <h4>${insight.title}</h4>
                <p>${insight.content}</p>
            `;
            insightsList.appendChild(insightItem);
        });
    }
} 