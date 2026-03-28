from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import cv2
import os
import numpy as np

app = Flask(__name__)
CORS(app)

DB_PATH = 'traffic.db'


# ================= DATABASE =================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')

    # Default admin
    cursor.execute('SELECT * FROM users WHERE username=?', ('admin',))
    if cursor.fetchone() is None:
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                       ('admin', generate_password_hash('1234')))
        conn.commit()

    conn.close()


# ================= AUTH APIs =================
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()

    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': 'Required fields missing'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                       (username, generate_password_hash(password)))
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'User already exists'}), 400
    finally:
        conn.close()

    return jsonify({'success': True})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    username = data.get('username')
    password = data.get('password')

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM users WHERE username=?', (username,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        return jsonify({'success': True, 'role': 'admin'})
    else:
        return jsonify({'success': False}), 401


# ================= VEHICLE DETECTION =================
def count_vehicles(video_path):
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return 0

    cap = cv2.VideoCapture(video_path)
    bg = cv2.createBackgroundSubtractorMOG2()

    count = 0
    frame_limit = 80   # faster processing

    for _ in range(frame_limit):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))

        fgmask = bg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 1500:
                count += 1

    cap.release()
    return count


# ================= YOLO DETECTION =================
YOLO_DIR = 'yolo'
YOLO_CFG = os.path.join(YOLO_DIR, 'yolov3.cfg')
YOLO_WEIGHTS = os.path.join(YOLO_DIR, 'yolov3.weights')
YOLO_NAMES = os.path.join(YOLO_DIR, 'coco.names')

yolo_net = None
yolo_classes = []


def load_yolo_model():
    global yolo_net, yolo_classes
    if yolo_net is not None:
        return True

    missing = []
    for p in (YOLO_CFG, YOLO_WEIGHTS, YOLO_NAMES):
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        return False

    with open(YOLO_NAMES, 'r') as f:
        yolo_classes = [line.strip() for line in f if line.strip()]

    yolo_net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
    yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return True


def yolo_detect(image, conf_threshold=0.5, nms_threshold=0.4):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)

    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
    outputs = yolo_net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_threshold:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x = int(center_x - bw / 2)
                y = int(center_y - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            cls = yolo_classes[class_ids[i]] if 0 <= class_ids[i] < len(yolo_classes) else 'unknown'
            x, y, bw, bh = boxes[i]
            results.append({
                'class': cls,
                'confidence': float(confidences[i]),
                'box': {'x': int(x), 'y': int(y), 'w': int(bw), 'h': int(bh)}
            })

    return results


@app.route('/api/yolo-status', methods=['GET'])
def yolo_status():
    found = load_yolo_model()
    return jsonify({'yolo_loaded': found, 'model_files': {'cfg': YOLO_CFG, 'weights': YOLO_WEIGHTS, 'names': YOLO_NAMES}})


@app.route('/api/yolo-detect', methods=['POST'])
def yolo_detect_route():
    if not load_yolo_model():
        return jsonify({'success': False, 'message': 'YOLO files missing. Create yolo/yolov3.cfg, yolo/yolov3.weights, yolo/coco.names'}), 500

    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image file provided'}), 400

    file = request.files['image']
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'success': False, 'message': 'Unable to decode image'}), 400

    results = yolo_detect(frame)
    return jsonify({'success': True, 'detections': results})


# ================= TRAFFIC API =================
@app.route('/api/traffic', methods=['GET'])
def traffic():
    roads = {
        "Road1": count_vehicles("videos/road1.mp4"),
        "Road2": count_vehicles("videos/road2.mp4"),
        "Road3": count_vehicles("videos/road3.mp4"),
        "Road4": count_vehicles("videos/road4.mp4")
    }

    # Find highest traffic
    green_road = max(roads, key=roads.get)

    # 🚑 Emergency logic (improved)
    emergency = False
    emergency_type = None

    # Example: if any road has extreme traffic
    for road, value in roads.items():
        if value > 70:
            emergency = True
            emergency_type = "Ambulance"
            green_road = road
            break

    return jsonify({
        "roads": roads,
        "green_road": green_road,
        "emergency": emergency,
        "type": emergency_type
    })


# ================= RUN APP =================
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=False)  # production-like server (no Flask debug mode)
