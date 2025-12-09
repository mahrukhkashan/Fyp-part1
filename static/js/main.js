/**
 * Main JavaScript file for Sepsis Prediction System
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    initTooltips();
    
    // Initialize charts if on dashboard
    if (document.getElementById('riskChart')) {
        initDashboardCharts();
    }
    
    // Initialize event listeners
    initEventListeners();
    
    // Check authentication status
    checkAuthStatus();
});

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize dashboard charts
 */
function initDashboardCharts() {
    // Risk Distribution Chart
    const riskCtx = document.getElementById('riskChart').getContext('2d');
    const riskChart = new Chart(riskCtx, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [65, 25, 10],
                backgroundColor: ['#28a745', '#ffc107', '#dc3545'],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });

    // Performance Metrics Chart
    const perfCtx = document.getElementById('performanceChart').getContext('2d');
    const perfChart = new Chart(perfCtx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity'],
            datasets: [{
                label: 'Model Performance',
                data: [85, 82, 88, 85, 90, 83],
                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                borderColor: 'rgba(52, 152, 219, 1)',
                pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(52, 152, 219, 1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 50,
                    suggestedMax: 100,
                    ticks: {
                        backdropColor: 'transparent'
                    }
                }
            }
        }
    });

    // Store charts for updates
    window.charts = {
        risk: riskChart,
        performance: perfChart
    };
}

/**
 * Initialize event listeners
 */
function initEventListeners() {
    // Prediction form submission
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }

    // Chat input enter key
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });
    }

    // Patient search
    const searchInput = document.getElementById('patientSearch');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(handlePatientSearch, 300));
    }

    // Logout button
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }

    // Export buttons
    const exportBtns = document.querySelectorAll('.export-btn');
    exportBtns.forEach(btn => {
        btn.addEventListener('click', handleExport);
    });

    // Alert acknowledge buttons
    const acknowledgeBtns = document.querySelectorAll('.acknowledge-alert');
    acknowledgeBtns.forEach(btn => {
        btn.addEventListener('click', handleAcknowledgeAlert);
    });
}

/**
 * Handle prediction form submission
 */
async function handlePredictionSubmit(e) {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // Validate data
    const errors = validatePredictionData(data);
    if (errors.length > 0) {
        showError(errors.join('\n'));
        return;
    }

    // Show loading
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Predicting...';
    submitBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.success) {
            displayPredictionResult(result);
        } else {
            showError(result.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        // Restore button
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
}

/**
 * Validate prediction data
 */
function validatePredictionData(data) {
    const errors = [];

    // Required fields
    const requiredFields = ['age', 'heart_rate', 'temperature', 'respiratory_rate'];
    requiredFields.forEach(field => {
        if (!data[field]) {
            errors.push(`${field.replace('_', ' ').toUpperCase()} is required`);
        }
    });

    // Numeric validation
    const numericFields = {
        'age': { min: 0, max: 120 },
        'heart_rate': { min: 30, max: 250 },
        'temperature': { min: 30, max: 45 },
        'respiratory_rate': { min: 8, max: 60 },
        'wbc': { min: 0, max: 100 },
        'lactate': { min: 0, max: 20 }
    };

    Object.entries(numericFields).forEach(([field, range]) => {
        if (data[field]) {
            const value = parseFloat(data[field]);
            if (isNaN(value)) {
                errors.push(`${field.replace('_', ' ').toUpperCase()} must be a number`);
            } else if (value < range.min || value > range.max) {
                errors.push(`${field.replace('_', ' ').toUpperCase()} must be between ${range.min} and ${range.max}`);
            }
        }
    });

    return errors;
}

/**
 * Display prediction result
 */
function displayPredictionResult(result) {
    const resultsContainer = document.getElementById('predictionResults');
    if (!resultsContainer) return;

    // Show results section
    resultsContainer.classList.remove('d-none');
    
    // Update risk percentage
    const riskPercentage = result.probability * 100;
    document.getElementById('riskPercentage').textContent = riskPercentage.toFixed(1) + '%';
    
    // Update risk badge
    const riskBadge = document.getElementById('riskBadge');
    riskBadge.textContent = result.risk_level;
    riskBadge.className = 'risk-badge ' + getRiskClass(result.risk_level);
    
    // Update key factors
    const keyFactorsList = document.getElementById('keyFactors');
    if (result.explanation && result.explanation.feature_effects) {
        keyFactorsList.innerHTML = '';
        const topFactors = result.explanation.feature_effects.slice(0, 3);
        
        topFactors.forEach(factor => {
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center';
            
            const factorName = factor.feature.replace(/_/g, ' ').toUpperCase();
            const factorImpact = Math.abs(factor.shap_value).toFixed(3);
            const direction = factor.shap_value > 0 ? 'Increases risk' : 'Decreases risk';
            
            li.innerHTML = `
                <div>
                    <strong>${factorName}</strong>
                    <div class="small text-muted">${direction}</div>
                </div>
                <span class="badge bg-primary rounded-pill">${factorImpact}</span>
            `;
            keyFactorsList.appendChild(li);
        });
    }
    
    // Update detailed explanation
    const detailedExplanation = document.getElementById('detailedExplanation');
    if (result.explanation) {
        let explanationHTML = `
            <p><strong>Base Value:</strong> ${result.explanation.base_value.toFixed(4)}</p>
            <p><strong>Prediction Confidence:</strong> ${(result.confidence || 0.85).toFixed(2)}</p>
        `;
        
        if (result.explanation.feature_effects) {
            explanationHTML += '<h6>Top Contributing Factors:</h6><ul class="mb-3">';
            result.explanation.feature_effects.slice(0, 5).forEach(factor => {
                const direction = factor.shap_value > 0 ? 'increases' : 'decreases';
                explanationHTML += `<li><strong>${factor.feature.replace(/_/g, ' ')}</strong>: ${direction} risk (impact: ${Math.abs(factor.shap_value).toFixed(4)})</li>`;
            });
            explanationHTML += '</ul>';
        }
        
        detailedExplanation.innerHTML = explanationHTML;
    }
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
    
    // Show notification
    showNotification(`Prediction completed: ${result.risk_level} risk detected`, 'success');
}

/**
 * Get CSS class for risk level
 */
function getRiskClass(riskLevel) {
    switch(riskLevel.toLowerCase()) {
        case 'high risk': return 'risk-high';
        case 'medium risk': return 'risk-medium';
        case 'low risk': return 'risk-low';
        default: return '';
    }
}

/**
 * Send chat message
 */
async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addChatMessage(message, 'user');
    input.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        const result = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add bot response
        if (result.success) {
            addChatMessage(result.response, 'bot');
            
            // Show suggestions if available
            if (result.suggestions && result.suggestions.length > 0) {
                showChatSuggestions(result.suggestions);
            }
        } else {
            addChatMessage('Sorry, I encountered an error. Please try again.', 'bot');
        }
    } catch (error) {
        removeTypingIndicator();
        addChatMessage('Network error. Please check your connection.', 'bot');
    }
}

/**
 * Add message to chat
 */
function addChatMessage(message, sender) {
    const container = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message fade-in`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    const senderName = sender === 'bot' ? 'Sepsis Assistant' : 'You';
    bubble.innerHTML = `<strong>${senderName}:</strong> ${message}`;
    
    messageDiv.appendChild(bubble);
    container.appendChild(messageDiv);
    
    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
}

/**
 * Show typing indicator
 */
function showTypingIndicator() {
    const container = document.getElementById('chatContainer');
    const indicator = document.createElement('div');
    indicator.className = 'chat-message bot-message typing-indicator';
    indicator.id = 'typingIndicator';
    indicator.innerHTML = `
        <div class="message-bubble">
            <strong>Sepsis Assistant:</strong>
            <span class="typing-dots">
                <span class="dot">.</span>
                <span class="dot">.</span>
                <span class="dot">.</span>
            </span>
        </div>
    `;
    container.appendChild(indicator);
    container.scrollTop = container.scrollHeight;
}

/**
 * Remove typing indicator
 */
function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

/**
 * Show chat suggestions
 */
function showChatSuggestions(suggestions) {
    const container = document.getElementById('chatContainer');
    const suggestionsDiv = document.createElement('div');
    suggestionsDiv.className = 'suggestions-container mt-3';
    
    let suggestionsHTML = '<div class="small text-muted mb-2">Quick suggestions:</div><div class="btn-group" role="group">';
    suggestions.slice(0, 3).forEach(suggestion => {
        suggestionsHTML += `<button type="button" class="btn btn-sm btn-outline-primary suggestion-btn">${suggestion}</button>`;
    });
    suggestionsHTML += '</div>';
    
    suggestionsDiv.innerHTML = suggestionsHTML;
    container.appendChild(suggestionsDiv);
    
    // Add click listeners to suggestion buttons
    const suggestionBtns = suggestionsDiv.querySelectorAll('.suggestion-btn');
    suggestionBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            document.getElementById('chatInput').value = this.textContent;
            sendChatMessage();
            suggestionsDiv.remove();
        });
    });
    
    container.scrollTop = container.scrollHeight;
}

/**
 * Handle patient search
 */
async function handlePatientSearch(e) {
    const searchTerm = e.target.value.trim();
    
    if (searchTerm.length < 2) {
        clearSearchResults();
        return;
    }
    
    try {
        const response = await fetch(`/api/patients/search?q=${encodeURIComponent(searchTerm)}`);
        const result = await response.json();
        
        if (result.success) {
            displaySearchResults(result.patients);
        }
    } catch (error) {
        console.error('Search error:', error);
    }
}

/**
 * Display search results
 */
function displaySearchResults(patients) {
    const resultsContainer = document.getElementById('searchResults');
    if (!resultsContainer) return;
    
    if (patients.length === 0) {
        resultsContainer.innerHTML = '<div class="text-center text-muted py-3">No patients found</div>';
        return;
    }
    
    let resultsHTML = '<div class="list-group">';
    patients.forEach(patient => {
        resultsHTML += `
            <a href="#" class="list-group-item list-group-item-action" data-patient-id="${patient.id}">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">${patient.name}</h6>
                    <small class="text-muted">ID: ${patient.id}</small>
                </div>
                <p class="mb-1">Age: ${patient.age} | Risk: <span class="${getRiskClass(patient.risk_level)}">${patient.risk_level}</span></p>
                <small class="text-muted">Last prediction: ${patient.last_prediction}</small>
            </a>
        `;
    });
    resultsHTML += '</div>';
    
    resultsContainer.innerHTML = resultsHTML;
    
    // Add click listeners
    const patientLinks = resultsContainer.querySelectorAll('.list-group-item');
    patientLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const patientId = this.getAttribute('data-patient-id');
            loadPatientDetails(patientId);
        });
    });
}

/**
 * Clear search results
 */
function clearSearchResults() {
    const resultsContainer = document.getElementById('searchResults');
    if (resultsContainer) {
        resultsContainer.innerHTML = '';
    }
}

/**
 * Load patient details
 */
async function loadPatientDetails(patientId) {
    try {
        const response = await fetch(`/api/patient/${patientId}`);
        const result = await response.json();
        
        if (result.success) {
            displayPatientDetails(result);
        } else {
            showError('Failed to load patient details');
        }
    } catch (error) {
        showError('Network error loading patient details');
    }
}

/**
 * Display patient details
 */
function displayPatientDetails(data) {
    // Implementation depends on your UI structure
    console.log('Patient details:', data);
    // You would update various UI elements with patient data here
}

/**
 * Handle logout
 */
async function handleLogout() {
    try {
        const response = await fetch('/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const result = await response.json();
        
        if (result.success) {
            window.location.href = '/login';
        }
    } catch (error) {
        showError('Logout failed: ' + error.message);
    }
}

/**
 * Handle export
 */
async function handleExport(e) {
    const exportType = this.getAttribute('data-export-type');
    
    try {
        const response = await fetch(`/api/export/${exportType}`);
        const result = await response.json();
        
        if (result.success) {
            // Create and download file
            const blob = new Blob([result.data], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = result.filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showNotification('Export completed successfully', 'success');
        }
    } catch (error) {
        showError('Export failed: ' + error.message);
    }
}

/**
 * Handle alert acknowledgement
 */
async function handleAcknowledgeAlert(e) {
    const alertId = this.getAttribute('data-alert-id');
    
    try {
        const response = await fetch(`/api/alerts/${alertId}/acknowledge`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const result = await response.json();
        
        if (result.success) {
            // Update UI
            const alertElement = this.closest('.alert');
            if (alertElement) {
                alertElement.classList.remove('alert-warning');
                alertElement.classList.add('alert-secondary');
                this.remove();
                
                const statusBadge = alertElement.querySelector('.alert-status');
                if (statusBadge) {
                    statusBadge.textContent = 'Acknowledged';
                }
            }
            
            showNotification('Alert acknowledged', 'success');
        }
    } catch (error) {
        showError('Failed to acknowledge alert');
    }
}

/**
 * Check authentication status
 */
async function checkAuthStatus() {
    try {
        const response = await fetch('/check_auth');
        const result = await response.json();
        
        if (!result.authenticated && !window.location.pathname.includes('/login')) {
            window.location.href = '/login';
        }
    } catch (error) {
        console.error('Auth check failed:', error);
    }
}

/**
 * Show error message
 */
function showError(message) {
    // Remove any existing error alerts
    const existingAlerts = document.querySelectorAll('.alert-error');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create new alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show alert-error';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page
    const container = document.querySelector('.container') || document.body;
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `toast align-items-center text-white bg-${type} border-0`;
    notification.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    // Add to container
    const container = document.getElementById('notificationContainer') || createNotificationContainer();
    container.appendChild(notification);
    
    // Initialize and show toast
    const toast = new bootstrap.Toast(notification, { delay: 3000 });
    toast.show();
    
    // Remove after hide
    notification.addEventListener('hidden.bs.toast', function () {
        notification.remove();
    });
}

/**
 * Create notification container if it doesn't exist
 */
function createNotificationContainer() {
    const container = document.createElement('div');
    container.id = 'notificationContainer';
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    document.body.appendChild(container);
    return container;
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Format date
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/**
 * Format number with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Get color for value based on thresholds
 */
function getValueColor(value, low, high) {
    if (value < low) return 'text-primary'; // Below normal (blue)
    if (value > high) return 'text-danger';  // Above normal (red)
    return 'text-success';                   // Normal (green)
}

/**
 * Update dashboard charts
 */
function updateCharts(data) {
    if (window.charts && window.charts.risk) {
        window.charts.risk.data.datasets[0].data = data.riskDistribution;
        window.charts.risk.update();
    }
    
    if (window.charts && window.charts.performance) {
        window.charts.performance.data.datasets[0].data = data.performanceMetrics;
        window.charts.performance.update();
    }
}

/**
 * Refresh dashboard data
 */
async function refreshDashboard() {
    try {
        const response = await fetch('/api/dashboard_stats');
        const result = await response.json();
        
        if (result.success) {
            // Update stats
            document.getElementById('totalPredictions').textContent = formatNumber(result.stats.total_predictions);
            document.getElementById('highRiskCases').textContent = formatNumber(result.stats.high_risk_cases);
            document.getElementById('modelAccuracy').textContent = result.stats.model_accuracy + '%';
            
            // Update charts
            updateCharts({
                riskDistribution: [
                    result.stats.low_risk_cases,
                    result.stats.medium_risk_cases,
                    result.stats.high_risk_cases
                ],
                performanceMetrics: [
                    result.stats.model_accuracy,
                    85, 88, 86, 90, 82 // Sample metrics
                ]
            });
        }
    } catch (error) {
        console.error('Failed to refresh dashboard:', error);
    }
}

// Auto-refresh dashboard every 30 seconds
if (window.location.pathname.includes('dashboard')) {
    setInterval(refreshDashboard, 30000);
}

// Export functions for use in HTML
window.sendChatMessage = sendChatMessage;
window.refreshDashboard = refreshDashboard;
window.loadPatientDetails = loadPatientDetails;