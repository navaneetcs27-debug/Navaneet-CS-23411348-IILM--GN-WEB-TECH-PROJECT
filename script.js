// 🔐 Protect page (login required)
if (localStorage.getItem("loggedIn") !== "true") {
    window.location.href = "login.html";
}
// 🔊 VOICE ALERT
const API_BASE = window.location.protocol === "file:"
    ? "http://127.0.0.1:5000/api"
    : `${window.location.origin}/api`;
function speakAlert(message) {
    let speech = new SpeechSynthesisUtterance(message);
    window.speechSynthesis.speak(speech);
}

let map;
let carMarkers = [];
let chart;
let alertShown = false;
let moderateAlertShown = false; // 🔥 prevent repeated moderate voice

// 🚦 Traffic Signal
function setSignal(color) {

    document.getElementById("red").classList.remove("active-red");
    document.getElementById("yellow").classList.remove("active-yellow");
    document.getElementById("green").classList.remove("active-green");

    if (color === "red") {
        document.getElementById("red").classList.add("active-red");
    }

    if (color === "yellow") {
        document.getElementById("yellow").classList.add("active-yellow");
    }

    if (color === "green") {
        document.getElementById("green").classList.add("active-green");
    }
}

// 🚑 Emergency
function activateEmergency(type) {
    addAlert("🚑 " + type + " Priority Activated", "info");
    speakAlert(type + " vehicle priority activated");
    setSignal("green");
}

// 📊 Road Data
function displayRoads(data) {
    let container = document.getElementById("roads-data");
    container.innerHTML = "";

    for (let road in data.roads) {
        let div = document.createElement("div");
        div.innerText = road + " → " + data.roads[road];

        if (road === data.green_road) {
            div.classList.add("active");
        }

        container.appendChild(div);
    }
}

// 🗺️ Map + Moving Cars
function loadMap(data) {

    if (!map) {
        map = L.map('map').setView([28.6139, 77.2090], 12); // Delhi

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap'
        }).addTo(map);
    }

    carMarkers.forEach(marker => map.removeLayer(marker));
    carMarkers = [];

    const routes = {
        "Road1": [[28.61, 77.20], [28.62, 77.21]],
        "Road2": [[28.60, 77.22], [28.63, 77.23]],
        "Road3": [[28.64, 77.24], [28.65, 77.25]],
        "Road4": [[28.66, 77.26], [28.67, 77.27]]
    };

    // Car icon
    const carIcon = L.divIcon({
        html: '🚗',
        className: 'car-marker',
        iconSize: [30, 30],
        iconAnchor: [15, 15],
    });

    for (let road in routes) {

        let color = road === data.green_road ? "green" : "blue";

        L.polyline(routes[road], {
            color: color,
            weight: 5
        }).addTo(map);

        // Add 3 cars per road
        for (let carNum = 0; carNum < 3; carNum++) {
            let marker = L.marker(routes[road][carNum % routes[road].length], { icon: carIcon }).addTo(map);
            carMarkers.push(marker);

            let currentIndex = carNum % routes[road].length;
            let direction = 1;
            let animationId = null;

            function animateCar() {
                let start = routes[road][currentIndex];
                let end = routes[road][currentIndex + direction];
                if (!end) {
                    direction *= -1;
                    end = routes[road][currentIndex + direction];
                    if (!end) return;
                }

                let steps = 50;
                let step = 0;
                let latDiff = (end[0] - start[0]) / steps;
                let lngDiff = (end[1] - start[1]) / steps;

                animationId = setInterval(() => {
                    step++;
                    let newLat = start[0] + latDiff * step;
                    let newLng = start[1] + lngDiff * step;
                    marker.setLatLng([newLat, newLng]);

                    if (step >= steps) {
                        clearInterval(animationId);
                        currentIndex += direction;
                        if (currentIndex >= routes[road].length - 1) {
                            direction = -1;
                            currentIndex = routes[road].length - 1;
                        } else if (currentIndex <= 0) {
                            direction = 1;
                            currentIndex = 0;
                        }
                        setTimeout(animateCar, 500 + carNum * 500); // Stagger animations
                    }
                }, 50);
            }
            animateCar();
        }
    }
}

// 📊 Chart
function loadChart(data) {
    const ctx = document.getElementById('trafficChart').getContext('2d');

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(data.roads),
            datasets: [{
                label: 'Vehicle Count',
                data: Object.values(data.roads),
                backgroundColor: ['#ff4d4d','#ffa500','#4CAF50','#2196F3']
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 2000 }
        }
    });
}

// 📡 ALERT PANEL
function addAlert(message, type) {
    let panel = document.getElementById("alerts-panel");

    let div = document.createElement("div");
    div.classList.add("alert");

    if (type === "danger") div.classList.add("alert-danger");
    if (type === "warning") div.classList.add("alert-warning");
    if (type === "info") div.classList.add("alert-info");

    div.innerText = message;

    panel.prepend(div);

    if (panel.children.length > 10) {
        panel.removeChild(panel.lastChild);
    }
}

// 🔔 SMART ALERTS + VOICE
function checkTrafficAlert(data) {
    let max = Math.max(...Object.values(data.roads));

    // 🚨 Heavy Traffic
    if (max > 70 && !alertShown) {
        addAlert("🚨 Heavy Traffic Detected!", "danger");
        speakAlert("Heavy traffic detected");
        alertShown = true;
        moderateAlertShown = false;
    }

    // Reset
    if (max <= 70) {
        alertShown = false;
    }

    // ⚠️ Moderate Traffic
    if (max > 40 && max <= 70 && !moderateAlertShown) {
        addAlert("⚠️ Moderate Traffic", "warning");
        speakAlert("Moderate traffic detected");
        moderateAlertShown = true;
    }

    if (max <= 40) {
        moderateAlertShown = false;
    }
}

// 📢 Status
function updateStatus(text) {
    document.getElementById("status").innerText = text;
}

// 🚔 Challan
function generateChallan() {

    let vehicles = ["DL01AB1234", "DL02XY5678", "DL03MN4321"];
    let violations = ["Red Light Jump", "Over Speeding", "No Helmet"];

    let vehicle = vehicles[Math.floor(Math.random() * vehicles.length)];
    let violation = violations[Math.floor(Math.random() * violations.length)];
    let fine = Math.floor(Math.random() * 1000) + 500;

    let challanHTML = `
        <div class="challan-item">
            <h4>🚔 E-Challan</h4>
            <p><b>Vehicle:</b> ${vehicle}</p>
            <p><b>Violation:</b> ${violation}</p>
            <p><b>Fine:</b> ₹${fine}</p>
            <hr>
        </div>
    `;

    let box = document.getElementById("challan-box");

    box.innerHTML = challanHTML + box.innerHTML;

    let items = box.getElementsByClassName("challan-item");
    if (items.length > 5) {
        box.removeChild(items[items.length - 1]);
    }

    addAlert("🚔 Challan Generated for " + vehicle, "info");
}

// 🔄 LIVE SYSTEM
function refreshData() {
    // Use static data - no backend needed
    const data = {
        "roads": {
            "Road1": 25,
            "Road2": 15,
            "Road3": 35,
            "Road4": 10
        },
        "green_road": "Road3",
        "emergency": false,
        "type": null
    };

    displayRoads(data);
    loadMap(data);
    loadChart(data);
    checkTrafficAlert(data);
    updateStatus("🚦 Green Signal → " + data.green_road);

    addAlert("📊 Data loaded (static mode)", "info");

    setSignal("red");
    setTimeout(() => setSignal("yellow"), 1000);
    setTimeout(() => setSignal("green"), 2000);

    if (Math.random() > 0.7) {
        generateChallan();
    }
}

// 🚀 START
function getFallbackTrafficData() {
    return {
        "roads": {
            "Road1": 25,
            "Road2": 15,
            "Road3": 35,
            "Road4": 10
        },
        "green_road": "Road3",
        "emergency": false,
        "type": null
    };
}

function applyTrafficData(data, modeLabel) {
    displayRoads(data);
    loadMap(data);
    loadChart(data);
    checkTrafficAlert(data);
    updateStatus("ðŸš¦ Green Signal â†’ " + data.green_road);
    addAlert(`ðŸ“Š Data loaded (${modeLabel})`, "info");

    setSignal("red");
    setTimeout(() => setSignal("yellow"), 1000);
    setTimeout(() => setSignal("green"), 2000);

    if (Math.random() > 0.7) {
        generateChallan();
    }
}

function refreshData() {
    fetch(`${API_BASE}/traffic`)
        .then(res => {
            if (!res.ok) {
                throw new Error(`Traffic API failed with status ${res.status}`);
            }
            return res.json();
        })
        .then(data => applyTrafficData(data, "live mode"))
        .catch(error => {
            console.warn("Traffic API unavailable, using fallback data:", error);
            applyTrafficData(getFallbackTrafficData(), "static mode");
        });
}

window.onload = function () {

    // 🔐 Check login FIRST
    if (localStorage.getItem("loggedIn") !== "true") {
        window.location.href = "login.html";
        return;
    }

    // 👤 Show user info
    let role = localStorage.getItem("role") || "user";
    let user = localStorage.getItem("username") || "guest";

    if (document.getElementById("user-role")) {
        document.getElementById("user-role").innerText =
            "Logged in as: " + user + " (" + role + ")";
    }

    // 🚫 Hide challan for officer
    if (role === "officer") {
        let box = document.getElementById("challan-box");
        if (box) box.style.display = "none";
    }

    // 🌙 Mode persistence
    if (localStorage.getItem("mode") === "dark") {
        document.body.classList.add("dark");
    }

    // 🚀 Start system
    refreshData();
    setInterval(refreshData, 5000);
};

function toggleMode() {
    document.body.classList.toggle("dark");
}
function downloadChallanPDF() {
    window.print();
}
function logout() {
    localStorage.clear();
    alert("Logged out successfully");
    window.location.href = "login.html";
}

// 👉 ADD HERE 👇
function showSection(id) {
    let sections = document.querySelectorAll(".section");
    sections.forEach(sec => sec.style.display = "none");
    document.getElementById(id).style.display = "block";
    
    // Fix map display when shown
    if (id === 'map-section' && map) {
        setTimeout(() => map.invalidateSize(), 100);
    }
}
// 🌙 DARK MODE (FULL WORKING)
const modeBtn = document.getElementById("modeBtn");
if (modeBtn) {
    modeBtn.addEventListener("click", function () {
        document.body.classList.toggle("dark");

        // Save mode
        if (document.body.classList.contains("dark")) {
            localStorage.setItem("mode", "dark");
        } else {
            localStorage.setItem("mode", "light");
        }
    });
} else {
    console.warn('modeBtn not found in DOM; dark mode toggle not attached');
}

// Load saved mode on refresh
window.addEventListener("load", function () {
    if (localStorage.getItem("mode") === "dark") {
        document.body.classList.add("dark");
    }
});
// ✅ FIXED TOGGLE (waits for page load)
window.addEventListener("DOMContentLoaded", function () {

    let toggleBtn = document.getElementById("menuToggle");
    let sidebar = document.querySelector(".sidebar");

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener("click", function () {
            sidebar.classList.toggle("collapsed");
        });
    }

});
function toggleMenu() {
    let sidebar = document.getElementById("sidebar");
    sidebar.classList.toggle("collapsed");
}
const API_BASE = window.location.protocol === "file:"
    ? "http://127.0.0.1:5000/api"
    : `${window.location.origin}/api`;
