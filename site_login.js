const SITE_ORIGIN = ["127.0.0.1", "localhost"].includes(window.location.hostname) || window.location.protocol === "file:"
    ? "http://127.0.0.1:5000"
    : window.location.origin;
const API_BASE = `${SITE_ORIGIN}/api`;

let currentOtp = null;
let otpExpire = null;

function setBackendStatus(message, ok = true) {
    const statusEl = document.getElementById("backend-status");
    if (!statusEl) return;
    statusEl.innerText = message;
    statusEl.style.color = ok ? "green" : "red";
}

async function readJsonSafe(res) {
    const text = await res.text();
    if (!text) return {};
    try {
        return JSON.parse(text);
    } catch (error) {
        throw new Error("Invalid server response");
    }
}

function checkBackend() {
    if (window.location.origin === SITE_ORIGIN) {
        setBackendStatus("Backend running", true);
        return;
    }

    fetch(`${SITE_ORIGIN}/healthz`)
        .then(async res => {
            if (!res.ok) throw new Error(`Status ${res.status}`);
            await readJsonSafe(res);
            setBackendStatus("Backend running", true);
        })
        .catch(() => {
            setBackendStatus(`Backend not reachable. Open ${SITE_ORIGIN}/login.html`, false);
        });
}

function login() {
    const user = document.getElementById("login-user").value.trim();
    const pass = document.getElementById("login-pass").value.trim();

    if (!user || !pass) {
        alert("Please enter username and password");
        return;
    }

    fetch(`${API_BASE}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: user, password: pass })
    })
        .then(async res => {
            const data = await readJsonSafe(res);
            if (!res.ok || !data.success) {
                throw new Error(data.message || "Invalid login");
            }
            return data;
        })
        .then(data => {
            localStorage.setItem("loggedIn", "true");
            localStorage.setItem("username", user);
            localStorage.setItem("role", data.role || "user");
            window.location.href = `${SITE_ORIGIN}/index.html`;
        })
        .catch(err => {
            alert(err.message);
        });
}

function signup() {
    const user = document.getElementById("signup-user").value.trim();
    const pass = document.getElementById("signup-pass").value.trim();

    if (!user || !pass) {
        alert("Enter username and password to sign up");
        return;
    }

    fetch(`${API_BASE}/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: user, password: pass })
    })
        .then(async res => {
            const data = await readJsonSafe(res);
            if (!res.ok || !data.success) {
                throw new Error(data.message || "Registration failed");
            }
            return data;
        })
        .then(() => {
            alert("Registration successful. Please login.");
            showLogin();
        })
        .catch(err => {
            alert(err.message);
        });
}

function sendOtp() {
    const otpUser = document.getElementById("otp-user").value.trim();
    if (!otpUser) {
        alert("Enter username for OTP login");
        return;
    }

    currentOtp = Math.floor(100000 + Math.random() * 900000).toString();
    otpExpire = Date.now() + 5 * 60 * 1000;
    alert(`OTP for ${otpUser} is ${currentOtp} (demo only)`);
    document.getElementById("otp-code").style.display = "block";
    document.getElementById("verify-otp-btn").style.display = "block";
}

function verifyOtp() {
    const code = document.getElementById("otp-code").value.trim();
    if (!code) {
        alert("Enter OTP to proceed");
        return;
    }

    if (!currentOtp || Date.now() > otpExpire) {
        alert("OTP expired. Request a new one.");
        resetOtpForm();
        return;
    }

    if (code !== currentOtp) {
        alert("Invalid OTP");
        return;
    }

    localStorage.setItem("loggedIn", "true");
    localStorage.setItem("username", document.getElementById("otp-user").value.trim() || "otp-user");
    localStorage.setItem("role", "user");
    window.location.href = `${SITE_ORIGIN}/index.html`;
}

function resetOtpForm() {
    currentOtp = null;
    otpExpire = null;
    document.getElementById("otp-code").value = "";
    document.getElementById("otp-code").style.display = "none";
    document.getElementById("verify-otp-btn").style.display = "none";
}

function resetPass() {
    alert("Password reset link sent (demo)");
}

function showSignup() {
    document.getElementById("login-form").style.display = "none";
    document.getElementById("signup-form").style.display = "block";
    document.getElementById("forgot-form").style.display = "none";
    document.getElementById("otp-form").style.display = "none";
    document.getElementById("form-title").innerText = "Sign Up";
}

function showForgot() {
    document.getElementById("login-form").style.display = "none";
    document.getElementById("signup-form").style.display = "none";
    document.getElementById("forgot-form").style.display = "block";
    document.getElementById("otp-form").style.display = "none";
    document.getElementById("form-title").innerText = "Reset Password";
}

function showLogin() {
    document.getElementById("login-form").style.display = "block";
    document.getElementById("signup-form").style.display = "none";
    document.getElementById("forgot-form").style.display = "none";
    document.getElementById("otp-form").style.display = "none";
    document.getElementById("form-title").innerText = "Login";
    resetOtpForm();
}

function showOtp() {
    document.getElementById("login-form").style.display = "none";
    document.getElementById("signup-form").style.display = "none";
    document.getElementById("forgot-form").style.display = "none";
    document.getElementById("otp-form").style.display = "block";
    document.getElementById("form-title").innerText = "OTP Login";
    resetOtpForm();
}

document.addEventListener("DOMContentLoaded", checkBackend);
