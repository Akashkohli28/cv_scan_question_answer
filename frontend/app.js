/**
 * Frontend JavaScript for CV Scan application
 * Handles file upload, candidate management, and RAG queries
 */

const API_BASE = 'http://localhost:5000/api';
let selectedCandidateId = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const candidateName = document.getElementById('candidateName');
const uploadBtn = document.getElementById('uploadBtn');
const uploadStatus = document.getElementById('uploadStatus');
const uploadLoading = document.getElementById('uploadLoading');
const candidatesList = document.getElementById('candidatesList');

const questionInput = document.getElementById('questionInput');
const queryBtn = document.getElementById('queryBtn');
const queryStatus = document.getElementById('queryStatus');
const queryLoading = document.getElementById('queryLoading');
const resultsSection = document.getElementById('resultsSection');
const answerBox = document.getElementById('answerBox');
const sourcesList = document.getElementById('sourcesList');
const confidenceLevel = document.getElementById('confidenceLevel');

const totalCandidatesEl = document.getElementById('totalCandidates');
const apiStatusEl = document.getElementById('apiStatus');

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = '#f0f4ff';
});
uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.background = '';
});
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = '';
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelected();
    }
});

fileInput.addEventListener('change', handleFileSelected);
uploadBtn.addEventListener('click', uploadCV);
queryBtn.addEventListener('click', queryQuestion);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus();
    loadCandidates();
    setInterval(checkAPIStatus, 30000); // Check every 30s
});

/**
 * Handle file selection
 */
function handleFileSelected() {
    const file = fileInput.files[0];
    if (file) {
        const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        if (!validTypes.includes(file.type)) {
            showStatus('uploadStatus', 'Only PDF and DOCX files are supported', 'error');
            fileInput.value = '';
            return;
        }
        if (file.size > 50 * 1024 * 1024) {
            showStatus('uploadStatus', 'File size must be under 50MB', 'error');
            fileInput.value = '';
            return;
        }
        uploadArea.style.borderColor = '#764ba2';
        uploadArea.textContent = `✓ ${file.name} selected`;
    }
}

/**
 * Upload CV file
 */
async function uploadCV() {
    const file = fileInput.files[0];
    if (!file) {
        showStatus('uploadStatus', 'Please select a file', 'error');
        return;
    }

    uploadBtn.disabled = true;
    uploadLoading.style.display = 'block';
    hideStatus('uploadStatus');

    try {
        const formData = new FormData();
        formData.append('file', file);
        if (candidateName.value) {
            formData.append('candidate_name', candidateName.value);
        }

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }

        showStatus('uploadStatus', 
            `✓ CV uploaded! Candidate: ${data.candidate_name}`, 
            'success');
        
        // Reset form
        fileInput.value = '';
        candidateName.value = '';
        uploadArea.style.borderColor = '#667eea';
        uploadArea.textContent = '📁 Drag & drop your CV here\nor click to browse\n(Supported: PDF, DOCX - Max 50MB)';
        
        // Refresh candidates list
        await loadCandidates();
    } catch (error) {
        showStatus('uploadStatus', `Error: ${error.message}`, 'error');
    } finally {
        uploadBtn.disabled = false;
        uploadLoading.style.display = 'none';
    }
}

/**
 * Delete a candidate
 */
async function deleteCandidate(candidateId, event) {
    event.stopPropagation();
    
    if (!confirm('Are you sure you want to delete this CV? This action cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/candidate/${candidateId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Delete failed');
        }
        
        showStatus('uploadStatus', `✓ ${data.candidate_name} deleted successfully`, 'success');
        
        // Deselect if it was selected
        if (selectedCandidateId === candidateId) {
            selectedCandidateId = null;
        }
        
        // Reload candidates list
        loadCandidates();
    } catch (error) {
        showStatus('uploadStatus', `Error deleting CV: ${error.message}`, 'error');
    }
}

/**
 * Load and display candidates
 */
async function loadCandidates() {
    try {
        const response = await fetch(`${API_BASE}/candidates`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to load candidates');
        }

        totalCandidatesEl.textContent = data.total;

        if (data.total === 0) {
            candidatesList.innerHTML = '<p style="color: #999; text-align: center;">No candidates yet</p>';
            return;
        }

        candidatesList.innerHTML = data.candidates.map(candidate => `
            <div class="candidate-item ${candidate.candidate_id === selectedCandidateId ? 'selected' : ''}" 
                 onclick="selectCandidate('${candidate.candidate_id}', '${candidate.name}')">
                <div class="candidate-item-content">
                    <div class="candidate-name">
                        <strong>${candidate.name}</strong>
                        <div style="font-size: 0.85em; color: #666;">
                            ${candidate.email || 'No email'}
                        </div>
                    </div>
                    <button class="btn btn-delete" onclick="deleteCandidate('${candidate.candidate_id}', event)">Delete</button>
                </div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading candidates:', error);
        candidatesList.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
}

/**
 * Select a candidate
 */
function selectCandidate(candidateId, candidateName) {
    selectedCandidateId = candidateId;
    document.querySelectorAll('.candidate-item').forEach(item => {
        item.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');
    questionInput.placeholder = `Ask a question about ${candidateName}...`;
}

/**
 * Query question to RAG system
 */
async function queryQuestion() {
    const question = questionInput.value.trim();
    if (!question) {
        showStatus('queryStatus', 'Please enter a question', 'error');
        return;
    }

    queryBtn.disabled = true;
    queryLoading.style.display = 'block';
    hideStatus('queryStatus');
    resultsSection.style.display = 'none';

    try {
        const payload = {
            question: question,
            top_k: 5
        };

        if (selectedCandidateId) {
            payload.candidate_id = selectedCandidateId;
        }

        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Query failed');
        }

        // Display results
        answerBox.innerHTML = `<strong>Answer:</strong><p>${data.answer}</p>`;
        
        sourcesList.innerHTML = data.sources.length > 0 
            ? data.sources.map(source => `
                <div class="source-item">
                    <strong>${source.candidate_name}</strong> - ${source.chunk_type}
                    <br><small>${source.section} (Relevance: ${(source.relevance * 100).toFixed(0)}%)</small>
                </div>
            `).join('')
            : '<p style="color: #999;">No sources found</p>';
        
        confidenceLevel.textContent = data.confidence.toUpperCase();
        confidenceLevel.style.color = 
            data.confidence === 'high' ? '#28a745' : 
            data.confidence === 'medium' ? '#ffc107' : '#dc3545';

        resultsSection.style.display = 'block';
        showStatus('queryStatus', '✓ Query processed successfully', 'success');
    } catch (error) {
        showStatus('queryStatus', `Error: ${error.message}`, 'error');
    } finally {
        queryBtn.disabled = false;
        queryLoading.style.display = 'none';
    }
}

/**
 * Check API status
 */
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE.replace('/api', '')}/health`);
        if (response.ok) {
            apiStatusEl.textContent = '✓ Online';
            apiStatusEl.style.color = '#28a745';
        } else {
            throw new Error('Unhealthy');
        }
    } catch (error) {
        apiStatusEl.textContent = '✗ Offline';
        apiStatusEl.style.color = '#dc3545';
    }
}

/**
 * Show status message
 */
function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = `status ${type}`;
    element.style.display = 'block';
}

/**
 * Hide status message
 */
function hideStatus(elementId) {
    const element = document.getElementById(elementId);
    element.style.display = 'none';
}

/**
 * Keyboard shortcuts
 */
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to submit query
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && questionInput === document.activeElement) {
        queryQuestion();
    }
});
