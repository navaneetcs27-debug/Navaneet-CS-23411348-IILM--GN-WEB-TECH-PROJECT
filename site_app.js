const SITE_API_BASE = ["127.0.0.1", "localhost"].includes(window.location.hostname) || window.location.protocol === "file:"
    ? "http://127.0.0.1:5000/api"
    : `${window.location.origin}/api`;

if (localStorage.getItem("loggedIn") !== "true") {
    window.location.href = "http://127.0.0.1:5000/login.html";
}

let map;
let chart;
let carMarkers = [];
let routeLines = [];
let alertShown = false;
let moderateAlertShown = false;

const ROUTES = {
    Road1: [[28.604, 77.186], [28.610, 77.200], [28.620, 77.216], [28.632, 77.228]],
    Road2: [[28.596, 77.208], [28.607, 77.224], [28.619, 77.238], [28.633, 77.248]],
    Road3: [[28.614, 77.170], [28.622, 77.189], [28.631, 77.209], [28.641, 77.230]],
    Road4: [[28.585, 77.190], [28.595, 77.207], [28.606, 77.222], [28.618, 77.240]]
};

const ROUTE_COLORS = {
    Road1: "#22c55e",
    Road2: "#38bdf8",
    Road3: "#f59e0b",
    Road4: "#f97316"
};

async function readJsonSafe(res) {
    const text = await res.text();
    if (!text) return {};
    try {
        return JSON.parse(text);
    } catch (error) {
        throw new Error("Invalid server response");
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

function displayRoads(data) {
    const container = document.getElementById("roads-data");
    container.innerHTML = "";

    Object.entries(data.roads).forEach(([road, count]) => {
        const div = document.createElement("div");
        div.innerText = `${road} -> ${count}`;
        if (road === data.green_road) div.classList.add("active");
        container.appendChild(div);
    });
}

function loadMap(data) {
    if (!map) {
        map = L.map("map", {
            zoomControl: false
        }).setView([28.6139, 77.2090], 12);

        L.control.zoom({
            position: "bottomright"
        }).addTo(map);

        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution: "OpenStreetMap, CARTO"
        }).addTo(map);
    }

    carMarkers.forEach(marker => map.removeLayer(marker));
    routeLines.forEach(line => map.removeLayer(line));
    carMarkers = [];
    routeLines = [];

    const bounds = [];
    let liveVehicleCount = 0;

    Object.entries(ROUTES).forEach(([road, points], roadIndex) => {
        const isPriority = road === data.green_road;
        const baseColor = ROUTE_COLORS[road] || "#38bdf8";
        const roadFlow = Math.max(1, Math.min(4, Math.ceil((data.roads[road] || 0) / 20) || 1));

        const routeLine = L.polyline(points, {
            color: isPriority ? "#a3e635" : baseColor,
            weight: isPriority ? 8 : 6,
            opacity: isPriority ? 1 : 0.72,
            lineCap: "round"
        }).addTo(map);

        routeLines.push(routeLine);
        bounds.push(...points);

        for (let vehicleIndex = 0; vehicleIndex < roadFlow; vehicleIndex += 1) {
            const marker = createVehicleMarker(baseColor, road, vehicleIndex);
            carMarkers.push(marker);
            liveVehicleCount += 1;
            animateVehicle(marker, points, vehicleIndex, roadIndex);
        }
    });

    if (bounds.length) {
        map.fitBounds(bounds, {
            padding: [30, 30]
        });
    }

    updateMapInsights(data, liveVehicleCount);
}

function createVehicleMarker(color, road, vehicleIndex) {
    const vehicleIcon = L.divIcon({
        className: "vehicle-icon-wrapper",
        html: `<div class="vehicle-icon" style="--vehicle-color:${color}"><span></span></div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11]
    });

    const marker = L.marker(ROUTES[road][0], { icon: vehicleIcon }).addTo(map);
    marker.bindTooltip(`${road} vehicle ${vehicleIndex + 1}`, {
        direction: "top",
        offset: [0, -10]
    });
    return marker;
}

function animateVehicle(marker, points, vehicleIndex, roadIndex) {
    const totalSegments = points.length - 1;
    let segmentIndex = (vehicleIndex + roadIndex) % totalSegments;
    let progress = (vehicleIndex * 0.22) % 1;
    let direction = 1;
    const speed = 0.012 + vehicleIndex * 0.0025;

    function tick() {
        const start = points[segmentIndex];
        const end = points[segmentIndex + direction];

        if (!end) {
            direction *= -1;
            segmentIndex = Math.max(0, Math.min(points.length - 2, segmentIndex));
        }

        progress += speed;
        if (progress >= 1) {
            progress = 0;
            segmentIndex += direction;

            if (segmentIndex >= points.length - 1) {
                direction = -1;
                segmentIndex = points.length - 2;
            } else if (segmentIndex < 0) {
                direction = 1;
                segmentIndex = 0;
            }
        }

        const from = points[segmentIndex];
        const to = points[segmentIndex + direction];
        if (!to) {
            requestAnimationFrame(tick);
            return;
        }

        const lat = from[0] + (to[0] - from[0]) * progress;
        const lng = from[1] + (to[1] - from[1]) * progress;
        marker.setLatLng([lat, lng]);
        requestAnimationFrame(tick);
    }

    requestAnimationFrame(tick);
}

function updateMapInsights(data, liveVehicleCount) {
    const priorityRoad = document.getElementById("map-green-road");
    const vehicleCount = document.getElementById("map-vehicle-count");
    const routePills = document.getElementById("route-pills");
    const legend = document.getElementById("map-legend");

    if (priorityRoad) priorityRoad.innerText = data.green_road;
    if (vehicleCount) vehicleCount.innerText = String(liveVehicleCount);

    if (routePills) {
        routePills.innerHTML = Object.entries(data.roads).map(([road, count]) => `
            <div class="route-pill ${road === data.green_road ? "is-priority" : ""}">
                <span class="route-pill-dot" style="background:${ROUTE_COLORS[road] || "#38bdf8"}"></span>
                <span>${road}</span>
                <strong>${count}</strong>
            </div>
        `).join("");
    }

    if (legend) {
        legend.innerHTML = Object.keys(ROUTES).map(road => `
            <div class="legend-row">
                <span class="legend-line" style="background:${ROUTE_COLORS[road] || "#38bdf8"}"></span>
                <span>${road}</span>
                <small>${road === data.green_road ? "Priority route" : "Running route"}</small>
            </div>
        `).join("");
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
            responsive: true
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

    if (max <= 70) alertShown = false;

    if (max > 40 && max <= 70 && !moderateAlertShown) {
        addAlert("Moderate traffic detected", "warning");
        speakAlert("Moderate traffic detected");
        moderateAlertShown = true;
    }

    if (max <= 40) moderateAlertShown = false;
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

    document.getElementById("challan-box").innerHTML = `
        <div class="challan-item">
            <h4>E-Challan</h4>
            <p><b>Vehicle:</b> ${vehicle}</p>
            <p><b>Violation:</b> ${violation}</p>
            <p><b>Fine:</b> Rs ${fine}</p>
        </div>
    ` + document.getElementById("challan-box").innerHTML;

    addAlert(`Challan generated for ${vehicle}`, "info");
}

function activateEmergency(type) {
    addAlert(`${type} priority activated`, "info");
    speakAlert(`${type} vehicle priority activated`);
    setSignal("green");
}

function getFallbackTrafficData() {
    return {
        roads: { Road1: 25, Road2: 15, Road3: 35, Road4: 10 },
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
    updateStatus(`Traffic green signal -> ${data.green_road}`);
    addAlert(`Traffic data loaded (${modeLabel})`, "info");

    setSignal("red");
    setTimeout(() => setSignal("yellow"), 1000);
    setTimeout(() => setSignal("green"), 2000);
}

function refreshData() {
    fetch(`${SITE_API_BASE}/traffic`)
        .then(async res => {
            if (!res.ok) throw new Error(`Status ${res.status}`);
            return readJsonSafe(res);
        })
        .then(data => applyTrafficData(data, "live mode"))
        .catch(() => applyTrafficData(getFallbackTrafficData(), "static mode"));
}

function showSection(id) {
    document.querySelectorAll(".section").forEach(sec => {
        sec.style.display = "none";
    });
    document.getElementById(id).style.display = "block";
    if (id === "map-section" && map) {
        setTimeout(() => map.invalidateSize(), 100);
    }
}

function toggleMode() {
    document.body.classList.toggle("dark");
    localStorage.setItem("mode", document.body.classList.contains("dark") ? "dark" : "light");
}

function downloadChallanPDF() {
    window.print();
}

function logout() {
    localStorage.clear();
    window.location.href = "http://127.0.0.1:5000/login.html";
}

function toggleMenu() {
    const sidebar = document.getElementById("sidebar");
    sidebar.classList.toggle("collapsed");
}

window.addEventListener("load", () => {
    const role = localStorage.getItem("role") || "user";
    const user = localStorage.getItem("username") || "guest";
    const userRole = document.getElementById("user-role");
    if (userRole) userRole.innerText = `Logged in as: ${user} (${role})`;

    if (localStorage.getItem("mode") === "dark") {
        document.body.classList.add("dark");
    }

    refreshData();
    setInterval(refreshData, 5000);
});
