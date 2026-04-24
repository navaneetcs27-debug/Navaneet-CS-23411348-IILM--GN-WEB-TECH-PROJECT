const API_BASE = window.location.protocol === 'file:'
    ? 'http://127.0.0.1:5000/api'
    : `${window.location.origin}/api`;
let currentOtp = null;
let otpUser = null;
let otpExpire = null;

function setBackendStatus(message, ok = true) {
    let statusEl = document.getElementById('backend-status');
    if (!statusEl) return;
    statusEl.innerText = message;
    statusEl.style.color = ok ? 'green' : 'red';
}

function checkBackend() {
    setBackendStatus('Checking backend...');
    fetch(`${API_BASE}/yolo-status`, { method: 'GET' })
    .then(res => {
        if (!res.ok) throw new Error(`Status ${res.status}`);
        return res.json();
    })
    .then(_ => {
        setBackendStatus('Backend running ✓', true);
    })
    .catch(error => {
        console.error('Backend check failed:', error);
        setBackendStatus('Backend not reachable. Start the Flask server or open the deployed site URL.', false);
    });
}

function login() {
    let user = document.getElementById("login-user").value.trim();
    let pass = document.getElementById("login-pass").value.trim();

    if (!user || !pass) {
        alert('Please enter username and password');
        return;
    }

    fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user, password: pass })
    })
    .then(res => {
        if (!res.ok) {
            return res.json().then(err => { throw new Error(err.message || 'Server error ' + res.status); });
        }
        return res.json();
    })
    .then(data => {
        if (data.success) {
            localStorage.setItem('loggedIn', 'true');
            localStorage.setItem('username', user);
            localStorage.setItem('role', data.role || 'user');
            window.location.href = 'index.html';
        } else {
            alert(data.message || 'Invalid login');
        }
    })
    .catch(err => {
        console.error('Login error:', err);
        alert('Server error: ' + err.message + '\nPlease ensure the backend is running or use the deployed site URL.');
    });
}

function sendOtp() {
    otpUser = document.getElementById("otp-user").value.trim();
    if (!otpUser) {
        alert("Enter username for OTP login");
        return;
    }

    currentOtp = Math.floor(100000 + Math.random() * 900000).toString();
    otpExpire = Date.now() + 5 * 60 * 1000; // 5 minutes

    alert(`OTP for ${otpUser} is ${currentOtp} (demo only)`);
    document.getElementById("otp-code").style.display = "block";
    document.getElementById("verify-otp-btn").style.display = "block";
}

function verifyOtp() {
    let code = document.getElementById("otp-code").value.trim();
    if (!code) {
        alert("Enter OTP to proceed");
        return;
    }

    if (!currentOtp || Date.now() > otpExpire) {
        alert("OTP expired. Request a new one.");
        resetOtpForm();
        return;
    }

    if (code === currentOtp) {
        localStorage.setItem("loggedIn", "true");
        window.location.href = "index.html";
    } else {
        alert("Invalid OTP");
    }
}

function resetOtpForm() {
    currentOtp = null;
    otpUser = null;
    otpExpire = null;
    document.getElementById("otp-code").value = "";
    document.getElementById("otp-code").style.display = "none";
    document.getElementById("verify-otp-btn").style.display = "none";
}

function signup() {
    let user = document.querySelector('#signup-form input[type=text]').value.trim();
    let pass = document.querySelector('#signup-form input[type=password]').value.trim();

    if (!user || !pass) {
        alert('Enter username and password to sign up');
        return;
    }

    fetch(`${API_BASE}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user, password: pass })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            alert('Registration successful! Please login.');
            showLogin();
        } else {
            alert(data.message || 'Registration failed');
        }
    })
    .catch(err => {
        console.error('Register error:', err);
        alert('Server error, try again later');
    });
}

function resetPass() {
    alert("Password Reset Link Sent (Demo)");
}

/* SWITCH FORMS */
function showSignup() {
    document.getElementById("login-form").style.display = "none";
    document.getElementById("signup-form").style.display = "block";
    document.getElementById("forgot-form").style.display = "none";
    document.getElementById("form-title").innerText = "📝 Sign Up";
}

function showForgot() {
    document.getElementById("login-form").style.display = "none";
    document.getElementById("signup-form").style.display = "none";
    document.getElementById("forgot-form").style.display = "block";
    document.getElementById("form-title").innerText = "🔑 Reset Password";
}

function showLogin() {
    document.getElementById("login-form").style.display = "block";
    document.getElementById("signup-form").style.display = "none";
    document.getElementById("forgot-form").style.display = "none";
    document.getElementById("otp-form").style.display = "none";
    document.getElementById("form-title").innerText = "🔐 Login";
    resetOtpForm();
}

function showOtp() {
    document.getElementById("login-form").style.display = "none";
    document.getElementById("signup-form").style.display = "none";
    document.getElementById("forgot-form").style.display = "none";
    document.getElementById("otp-form").style.display = "block";
    document.getElementById("form-title").innerText = "🔑 OTP Login";
    resetOtpForm();
}

document.addEventListener('DOMContentLoaded', checkBackend);
