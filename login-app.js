const isLocalBrowser = ["127.0.0.1", "localhost"].includes(window.location.hostname);
const ORIGIN_BASE = window.location.protocol === "file:" || isLocalBrowser
    ? "http://127.0.0.1:5000"
    : window.location.origin;
const API_BASE = `${ORIGIN_BASE}/api`;

let currentOtp = null;
let otpUser = null;
let otpExpire = null;

async function readJsonSafe(res) {
    const text = await res.text();
    if (!text) {
        return {};
    }

    try {
        return JSON.parse(text);
    } catch (error) {
        throw new Error(`Server returned invalid JSON (${res.status})`);
    }
}

function setBackendStatus(message, ok = true) {
    const statusEl = document.getElementById("backend-status");
    if (!statusEl) return;
    statusEl.innerText = message;
    statusEl.style.color = ok ? "green" : "red";
}

function checkBackend() {
    setBackendStatus("Checking backend...");

    if (window.location.protocol !== "file:" && window.location.origin === ORIGIN_BASE) {
        setBackendStatus("Backend running", true);
        return;
    }

    fetch(`${ORIGIN_BASE}/healthz`, { method: "GET" })
        .then(async res => {
            if (!res.ok) throw new Error(`Status ${res.status}`);
            return readJsonSafe(res);
        })
        .then(() => {
            if (window.location.protocol === "file:") {
                setBackendStatus("Backend running. Opening app through Flask...", true);
                window.location.href = `${ORIGIN_BASE}/login.html`;
                return;
            }

            setBackendStatus("Backend running", true);
        })
        .catch(error => {
            console.error("Backend check failed:", error);

            if (window.location.protocol === "file:") {
                setBackendStatus(`Backend not reachable. Start Flask, then open ${ORIGIN_BASE}/login.html`, false);
                return;
            }

            setBackendStatus("Backend not reachable. Start the Flask server or open the deployed site URL.", false);
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

            if (!res.ok) {
                throw new Error(data.message || "Server error " + res.status);
            }

            return data;
        })
        .then(data => {
            if (!data.success) {
                alert(data.message || "Invalid login");
                return;
            }

            localStorage.setItem("loggedIn", "true");
            localStorage.setItem("username", user);
            localStorage.setItem("role", data.role || "user");
            window.location.href = "index.html";
        })
        .catch(err => {
            console.error("Login error:", err);
            alert("Server error: " + err.message + "\nOpen the app from http://127.0.0.1:5000/login.html");
        });
}

function sendOtp() {
    otpUser = document.getElementById("otp-user").value.trim();
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
    const user = document.querySelector("#signup-form input[type=text]").value.trim();
    const pass = document.querySelector("#signup-form input[type=password]").value.trim();

    if (!user || !pass) {
        alert("Enter username and password to sign up");
        return;
    }

    fetch(`${API_BASE}/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: user, password: pass })
    })
        .then(readJsonSafe)
        .then(data => {
            if (data.success) {
                alert("Registration successful! Please login.");
                showLogin();
            } else {
                alert(data.message || "Registration failed");
            }
        })
        .catch(err => {
            console.error("Register error:", err);
            alert("Server error, try again later");
        });
}

function resetPass() {
    alert("Password Reset Link Sent (Demo)");
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
