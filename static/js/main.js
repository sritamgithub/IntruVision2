// Helper function to show flash messages
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add alert to the top of the container
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 150);
    }, 5000);
}

// Handle auto-dismiss for flash messages
document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss flash messages after 5 seconds
    setTimeout(() => {
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(alert => {
            const bsAlert = new bootstrap.Alert(alert);
            setTimeout(() => bsAlert.close(), 5000);
        });
    }, 100);
    
    // Add active class to current nav item
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath) {
            link.classList.add('active');
        }
    });
});

// Face Recognition Video Stream Control
function toggleStream(btn) {
    const videoContainer = document.querySelector('.video-container');
    
    if (btn.dataset.status === 'stopped') {
        // Start stream
        btn.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
        btn.classList.remove('btn-success');
        btn.classList.add('btn-danger');
        btn.dataset.status = 'running';
        
        // Update UI
        videoContainer.innerHTML = '<img src="/video_feed" class="img-fluid w-100">';
        
        // Add live badge
        const badge = document.createElement('div');
        badge.className = 'video-overlay d-flex justify-content-center align-items-center';
        badge.innerHTML = '<span class="badge bg-success"><i class="fas fa-circle pulse"></i> Live</span>';
        videoContainer.appendChild(badge);
    } else {
        // Stop stream
        btn.innerHTML = '<i class="fas fa-play"></i> Start Camera';
        btn.classList.remove('btn-danger');
        btn.classList.add('btn-success');
        btn.dataset.status = 'stopped';
        
        // Update UI
        videoContainer.innerHTML = `
            <div class="camera-placeholder d-flex justify-content-center align-items-center">
                <div class="text-center">
                    <i class="fas fa-camera fa-5x text-muted mb-3"></i>
                    <p>Camera feed is not active</p>
                </div>
            </div>
        `;
    }
}

// Format dates nicely
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

// Confirm deletion
function confirmDelete(message, formId) {
    if (confirm(message)) {
        document.getElementById(formId).submit();
    }
    return false;
}

// Add custom file input behavior
document.addEventListener('DOMContentLoaded', function() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const fileName = this.files.length > 1 
                ? this.files.length + ' files selected' 
                : this.files[0]?.name || 'No file chosen';
            
            const label = this.nextElementSibling;
            if (label && label.classList.contains('custom-file-label')) {
                label.textContent = fileName;
            }
        });
    });
});