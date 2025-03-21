/**
 * Pet Video Dialogue Generator
 * Main Application Script
 */

// DOM Elements
const uploadForm = document.getElementById('upload-form');
const videoFileInput = document.getElementById('video-file');
const animalTypeSelect = document.getElementById('animal-type');
const personalitySelect = document.getElementById('personality');
const displayModeSelect = document.getElementById('display-mode');
const uploadSection = document.getElementById('upload-section');
const processingSection = document.getElementById('processing-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const progressBar = document.getElementById('progress-bar');
const taskIdSpan = document.getElementById('task-id');
const taskStatusSpan = document.getElementById('task-status');
const elapsedTimeSpan = document.getElementById('elapsed-time');
const cancelBtn = document.getElementById('cancel-btn');
const resultVideo = document.getElementById('result-video');
const resultSource = document.getElementById('result-source');
const resultAnimalType = document.getElementById('result-animal-type');
const resultPersonality = document.getElementById('result-personality');
const resultDisplayMode = document.getElementById('result-display-mode');
const downloadBtn = document.getElementById('download-btn');
const processNewBtn = document.getElementById('process-new-btn');
const retryBtn = document.getElementById('retry-btn');
const errorMessage = document.getElementById('error-message');

// State
let currentTask = null;
let statusInterval = null;
let elapsedTimeInterval = null;
let startTime = null;

// Initialize
function init() {
    // Add event listeners
    uploadForm.addEventListener('submit', handleFormSubmit);
    cancelBtn.addEventListener('click', handleCancel);
    processNewBtn.addEventListener('click', resetForm);
    retryBtn.addEventListener('click', resetForm);
    
    // Check for file size before submission
    videoFileInput.addEventListener('change', validateFileSize);
    
    // Add toggle for developer options
    const toggleDevBtn = document.getElementById('toggle-dev-options');
    if (toggleDevBtn) {
        toggleDevBtn.addEventListener('click', function() {
            const devOptions = document.getElementById('developer-options');
            if (devOptions) {
                devOptions.style.display = devOptions.style.display === 'none' ? 'block' : 'none';
            }
        });
    }
}

// Validate file size
function validateFileSize() {
    const file = videoFileInput.files[0];
    if (file) {
        // Max size: 100MB
        const maxSize = 100 * 1024 * 1024;
        if (file.size > maxSize) {
            alert('File is too large. Maximum file size is 100MB.');
            videoFileInput.value = '';
        }
    }
}

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Validate form
    if (!validateForm()) {
        return;
    }
    
    // Show processing section
    showSection('processing');
    
    // Reset progress
    progressBar.style.width = '0%';
    startTime = Date.now();
    
    // Upload the video
    try {
        const result = await uploadVideo();
        
        if (result && result.task_id) {
            currentTask = result.task_id;
            taskIdSpan.textContent = currentTask;
            
            // Start polling for status
            startStatusPolling();
            startElapsedTimeCounter();
        } else {
            throw new Error('Invalid response from server');
        }
    } catch (error) {
        showError(error.message || 'Failed to upload video');
    }
}

// Validate form
function validateForm() {
    if (!videoFileInput.files || videoFileInput.files.length === 0) {
        alert('Please select a video file');
        return false;
    }
    
    return true;
}

// Upload video
async function uploadVideo() {
    const formData = new FormData();
    formData.append('video', videoFileInput.files[0]);
    formData.append('animal_type', animalTypeSelect.value);
    formData.append('personality', personalitySelect.value);
    formData.append('display_mode', displayModeSelect.value);
    
    // Add forced state from developer options if selected
    const forcedStateSelect = document.getElementById('forced-state');
    if (forcedStateSelect && forcedStateSelect.value) {
        formData.append('forced_state', forcedStateSelect.value);
        console.log('Using forced state:', forcedStateSelect.value);
    }
    
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed: ${response.status} ${errorText}`);
    }
    
    return response.json();
}

// Start polling for status
function startStatusPolling() {
    if (statusInterval) {
        clearInterval(statusInterval);
    }
    
    statusInterval = setInterval(async () => {
        try {
            const status = await checkStatus();
            
            if (status.status === 'complete') {
                clearInterval(statusInterval);
                handleProcessingComplete(status);
            } else if (status.status === 'failed') {
                clearInterval(statusInterval);
                showError(status.error || 'Processing failed');
            } else if (status.status === 'processing') {
                updateProgress(status);
            }
        } catch (error) {
            clearInterval(statusInterval);
            showError(error.message || 'Failed to check status');
        }
    }, 1000);
}

// Start elapsed time counter
function startElapsedTimeCounter() {
    if (elapsedTimeInterval) {
        clearInterval(elapsedTimeInterval);
    }
    
    elapsedTimeInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        elapsedTimeSpan.textContent = formatTime(elapsed);
    }, 1000);
}

// Format time in seconds to MM:SS
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Check processing status
async function checkStatus() {
    if (!currentTask) {
        throw new Error('No active task');
    }
    
    const response = await fetch(`/status/${currentTask}`);
    
    if (!response.ok) {
        throw new Error(`Status check failed: ${response.status}`);
    }
    
    return response.json();
}

// Update progress bar
function updateProgress(status) {
    const progress = status.progress || 0;
    progressBar.style.width = `${progress}%`;
    taskStatusSpan.textContent = status.status || 'Processing';
}

// Handle processing complete
function handleProcessingComplete(status) {
    // Stop interval timers
    clearInterval(statusInterval);
    clearInterval(elapsedTimeInterval);
    
    // Set result details
    resultAnimalType.textContent = capitalizeFirstLetter(status.result.animal_type || 'auto');
    resultPersonality.textContent = capitalizeFirstLetter(status.result.personality || 'auto');
    resultDisplayMode.textContent = formatDisplayMode(status.result.display_mode || 'speech_bubble');
    
    // Show results section
    showSection('results');
    
    // Prepare the video container properly
    const videoContainer = document.getElementById('video-container');
    if (videoContainer) {
        // Clear everything
        videoContainer.innerHTML = '';
        // Force layout refresh by reading and then setting a property
        void videoContainer.offsetHeight;
        // Remove any classes that might cause issues
        videoContainer.className = 'mt-3 mb-3';
        // Make sure the container is visible
        videoContainer.style.display = 'block';
    }
    
    // Add a longer delay before creating the video to ensure DOM is fully updated
    setTimeout(() => {
        // Set video source with double cache-busting parameter
        const cacheBuster = new Date().getTime();
        const videoUrl = `${status.download_url}?v=${cacheBuster}`;
        
        // Use the createFreshVideo function from index.html
        if (window.createFreshVideo) {
            // Create the video and ensure we have a reference to it
            const videoElement = window.createFreshVideo(videoUrl);
            
            // Add an extra event handler to ensure the video is visible
            if (videoElement) {
                videoElement.addEventListener('loadedmetadata', function() {
                    // Force a refresh of the container's display
                    videoContainer.style.visibility = 'hidden';
                    setTimeout(() => {
                        videoContainer.style.visibility = 'visible';
                    }, 50);
                });
            }
        } else {
            console.error('createFreshVideo function not found');
            // Fallback to direct link if the function isn't available
            showError('Video display not available. Please use the download button.');
        }
        
        // Set download button with cache buster
        downloadBtn.href = videoUrl;
        downloadBtn.download = `pet_dialogue_${currentTask}.mp4`;
    }, 200); // Increased delay to ensure DOM is fully updated
}

// Handle cancel button
async function handleCancel() {
    if (!currentTask) {
        return;
    }
    
    // Stop polling
    clearInterval(statusInterval);
    clearInterval(elapsedTimeInterval);
    
    try {
        // Send delete request
        const response = await fetch(`/task/${currentTask}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`Cancel failed: ${response.status}`);
        }
        
        // Reset form
        resetForm();
    } catch (error) {
        showError(error.message || 'Failed to cancel processing');
    }
}

// Show error
function showError(message) {
    // Stop polling
    clearInterval(statusInterval);
    clearInterval(elapsedTimeInterval);
    
    // Set error message
    errorMessage.textContent = message;
    
    // Show error section
    showSection('error');
}

// Reset form to initial state
function resetForm() {
    // Reset form fields
    uploadForm.reset();
    
    // Clear current task
    currentTask = null;
    
    // Stop intervals
    clearInterval(statusInterval);
    clearInterval(elapsedTimeInterval);
    
    // Show upload section
    showSection('upload');
}

// Show a specific section and hide others
function showSection(section) {
    uploadSection.style.display = section === 'upload' ? 'block' : 'none';
    processingSection.style.display = section === 'processing' ? 'block' : 'none';
    resultsSection.style.display = section === 'results' ? 'block' : 'none';
    errorSection.style.display = section === 'error' ? 'block' : 'none';
    
    // If showing results, make sure video container is properly sized
    if (section === 'results') {
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
        }, 100);
    }
}

// Helper functions
function capitalizeFirstLetter(string) {
    if (!string) return '';
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function formatDisplayMode(mode) {
    if (!mode) return '';
    return mode.replace('_', ' ').split(' ').map(capitalizeFirstLetter).join(' ');
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
