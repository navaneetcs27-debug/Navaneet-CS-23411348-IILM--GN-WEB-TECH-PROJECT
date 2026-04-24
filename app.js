if (localStorage.getItem("loggedIn") !== "true") {
    window.location.href = "login.html";
}

const isLocalBrowser = ["127.0.0.1", "localhost"].includes(window.location.hostname);
const API_BASE = window.location.protocol === "file:" || isLocalBrowser
    ? "http://127.0.0.1:5000/api"
    : `${window.location.origin}/api`;

let map;
let carMarkers = [];
let chart;
let alertShown = false;
let moderateAlertShown = false;

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

function speakAlert(message) {
    const speech = new SpeechSynthesisUtterance(message);
    window.speechSynthesis.speak(speech);
}

function setSignal(color) {
    document.getElementById("red").classList.remove("active-red");
    document.getElementById("yellow").classList.remove("active-yellow");
    document.getElementById("green").classList.remove("active-green");

    if (color === "red") document.getElementById("red").classList.add("active-red");
    if (color === "yellow") document.getElementById("yellow").classList.add("active-yellow");
    if (color === "green") document.getElementById("green").classList.add("active-green");
}

function activateEmergency(type) {
    addAlert(type + " priority activated", "info");
    speakAlert(type + " vehicle priority activated");
    setSignal("green");
}

function displayRoads(data) {
    const container = document.getElementById("roads-data");
    container.innerHTML = "";

    for (const road in data.roads) {
        const div = document.createElement("div");
        div.innerText = road + " -> " + data.roads[road];

        if (road === data.green_road) {
            div.classList.add("active");
        }

        container.appendChild(div);
    }
}

function loadMap(data) {
    if (!map) {
        map = L.map("map").setView([28.6139, 77.2090], 12);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "OpenStreetMap"
        }).addTo(map);
    }

    carMarkers.forEach(marker => map.removeLayer(marker));
    carMarkers = [];

    const routes = {
        Road1: [[28.61, 77.20], [28.62, 77.21]],
        Road2: [[28.60, 77.22], [28.63, 77.23]],
        Road3: [[28.64, 77.24], [28.65, 77.25]],
        Road4: [[28.66, 77.26], [28.67, 77.27]]
    };

    const carIcon = L.divIcon({
        html: "CAR",
        className: "car-marker",
        iconSize: [30, 30],
        iconAnchor: [15, 15]
    });

    for (const road in routes) {
        const color = road === data.green_road ? "green" : "blue";

        L.polyline(routes[road], {
            color,
            weight: 5
        }).addTo(map);

        for (let carNum = 0; carNum < 3; carNum++) {
            const marker = L.marker(routes[road][carNum % routes[road].length], { icon: carIcon }).addTo(map);
            carMarkers.push(marker);

            let currentIndex = carNum % routes[road].length;
            let direction = 1;

            function animateCar() {
                const start = routes[road][currentIndex];
                let end = routes[road][currentIndex + direction];

                if (!end) {
                    direction *= -1;
                    end = routes[road][currentIndex + direction];
                    if (!end) return;
                }

                const steps = 50;
                let step = 0;
                const latDiff = (end[0] - start[0]) / steps;
                const lngDiff = (end[1] - start[1]) / steps;

                const animationId = setInterval(() => {
                    step += 1;
                    marker.setLatLng([
                        start[0] + latDiff * step,
                        start[1] + lngDiff * step
                    ]);

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

                        setTimeout(animateCar, 500 + carNum * 500);
                    }
                }, 50);
            }

            animateCar();
        }
    }
}

function loadChart(data) {
    const ctx = document.getElementById("trafficChart").getContext("2d");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: Object.keys(data.roads),
            datasets: [{
                label: "Vehicle Count",
                data: Object.values(data.roads),
                backgroundColor: ["#ff4d4d", "#ffa500", "#4CAF50", "#2196F3"]
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 2000 }
        }
    });
}

function addAlert(message, type) {
    const panel = document.getElementById("alerts-panel");
    const div = document.createElement("div");
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

function checkTrafficAlert(data) {
    const max = Math.max(...Object.values(data.roads));

    if (max > 70 && !alertShown) {
        addAlert("Heavy traffic detected", "danger");
        speakAlert("Heavy traffic detected");
        alertShown = true;
        moderateAlertShown = false;
    }

    if (max <= 70) {
        alertShown = false;
    }

    if (max > 40 && max <= 70 && !moderateAlertShown) {
        addAlert("Moderate traffic detected", "warning");
        speakAlert("Moderate traffic detected");
        moderateAlertShown = true;
    }

    if (max <= 40) {
        moderateAlertShown = false;
    }
}

function updateStatus(text) {
    document.getElementById("status").innerText = text;
}

function generateChallan() {
    const vehicles = ["DL01AB1234", "DL02XY5678", "DL03MN4321"];
    const violations = ["Red Light Jump", "Over Speeding", "No Helmet"];

    const vehicle = vehicles[Math.floor(Math.random() * vehicles.length)];
    const violation = violations[Math.floor(Math.random() * violations.length)];
    const fine = Math.floor(Math.random() * 1000) + 500;

    const challanHTML = `
        <div class="challan-item">
            <h4>E-Challan</h4>
            <p><b>Vehicle:</b> ${vehicle}</p>
            <p><b>Violation:</b> ${violation}</p>
            <p><b>Fine:</b> Rs ${fine}</p>
            <hr>
        </div>
    `;

    const box = document.getElementById("challan-box");
    box.innerHTML = challanHTML + box.innerHTML;

    const items = box.getElementsByClassName("challan-item");
    if (items.length > 5) {
        box.removeChild(items[items.length - 1]);
    }

    addAlert("Challan generated for " + vehicle, "info");
}

function getFallbackTrafficData() {
    return {
        roads: {
            Road1: 25,
            Road2: 15,
            Road3: 35,
            Road4: 10
        },
        green_road: "Road3",
        emergency: false,
        type: null
    };
}

function applyTrafficData(data, modeLabel) {
    displayRoads(data);
    loadMap(data);
    loadChart(data);
    checkTrafficAlert(data);
    updateStatus("Traffic green signal -> " + data.green_road);
    addAlert(`Traffic data loaded (${modeLabel})`, "info");

    setSignal("red");
    setTimeout(() => setSignal("yellow"), 1000);
    setTimeout(() => setSignal("green"), 2000);

    if (Math.random() > 0.7) {
        generateChallan();
    }
}

function refreshData() {
    fetch(`${API_BASE}/traffic`)
        .then(async res => {
            if (!res.ok) {
                throw new Error(`Traffic API failed with status ${res.status}`);
            }
            return readJsonSafe(res);
        })
        .then(data => applyTrafficData(data, "live mode"))
        .catch(error => {
            console.warn("Traffic API unavailable, using fallback data:", error);
            applyTrafficData(getFallbackTrafficData(), "static mode");
        });
}

window.onload = function () {
    if (localStorage.getItem("loggedIn") !== "true") {
        window.location.href = "login.html";
        return;
    }

    const role = localStorage.getItem("role") || "user";
    const user = localStorage.getItem("username") || "guest";

    if (document.getElementById("user-role")) {
        document.getElementById("user-role").innerText =
            "Logged in as: " + user + " (" + role + ")";
    }

    if (role === "officer") {
        const box = document.getElementById("challan-box");
        if (box) box.style.display = "none";
    }

    if (localStorage.getItem("mode") === "dark") {
        document.body.classList.add("dark");
    }

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

function showSection(id) {
    const sections = document.querySelectorAll(".section");
    sections.forEach(sec => {
        sec.style.display = "none";
    });

    document.getElementById(id).style.display = "block";

    if (id === "map-section" && map) {
        setTimeout(() => map.invalidateSize(), 100);
    }
}

const modeBtn = document.getElementById("modeBtn");
if (modeBtn) {
    modeBtn.addEventListener("click", function () {
        document.body.classList.toggle("dark");

        if (document.body.classList.contains("dark")) {
            localStorage.setItem("mode", "dark");
        } else {
            localStorage.setItem("mode", "light");
        }
    });
}

window.addEventListener("load", function () {
    if (localStorage.getItem("mode") === "dark") {
        document.body.classList.add("dark");
    }
});

window.addEventListener("DOMContentLoaded", function () {
    const toggleBtn = document.getElementById("menuToggle");
    const sidebar = document.querySelector(".sidebar");

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener("click", function () {
            sidebar.classList.toggle("collapsed");
        });
    }
});

function toggleMenu() {
    const sidebar = document.getElementById("sidebar");
    sidebar.classList.toggle("collapsed");
}
