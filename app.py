import os
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sklearn.metrics import precision_score, recall_score, f1_score
import pymysql
import cv2
import numpy as np
import json

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///traffic.db'  # Modify for MySQL/PostgreSQL
db = SQLAlchemy(app)
migrate = Migrate(app, db) 

# MySQL Connection
db = pymysql.connect(host="localhost", user="root", password="trevor", database="traffic_db")
cursor = db.cursor()

# Set upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Haar Cascade model for car detection
car_cascade = cv2.CascadeClassifier('cars.xml')

# Global variables to store evaluation metrics
global_precision = 0
global_recall = 0
global_f1 = 0

# Function: Control Traffic Signal
def control_traffic_signal(car_count_lane1, car_count_lane2):
    if car_count_lane1 > car_count_lane2:
        print("Lane 1 is congested. Redirecting cars to Lane 2.")
        return "Lane 2: Green, Lane 1: Red", "Consider using Lane 2 for faster travel."
    else:
        print("Lane 2 is congested. Redirecting cars to Lane 1.")
        return "Lane 1: Green, Lane 2: Red", "Consider using Lane 1 for faster travel."

# Function: Detect Cars in a Frame
def detect_cars(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    detected_boxes = []
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detected_boxes.append([x, y, x + w, y + h])  # Format: [x1, y1, x2, y2]

    return frame, detected_boxes, len(cars)

# Route: Video Feed with Real-Time Car Detection
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global global_precision, global_recall, global_f1
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')
        if not os.path.exists(video_path):
            video_path = 'traffic_video.mp4'  # Default video file

        camera = cv2.VideoCapture(video_path)

        if not camera.isOpened():
            print("Error: Could not open video file.")
            return

        # Create a ground truth data structure (simplified for demo)
        # In a real implementation, you would load this from a file
        ground_truth_data = []
        
        frame_count = 0
        all_detected_boxes = []

        while True:
            success, frame = camera.read()
            if not success:
                break

            # Detect cars in the frame
            frame, detected_boxes, car_count_lane1 = detect_cars(frame)
            all_detected_boxes.extend(detected_boxes)
            
            car_count_lane2 = car_count_lane1 // 2  # Simulated for lane 2
            signal_status, route_recommendation = control_traffic_signal(car_count_lane1, car_count_lane2)

            # Save data to database
            try:
                cursor.execute("INSERT INTO traffic_data (lane1_cars, lane2_cars, signal_status, route_recommendation) VALUES (%s, %s, %s, %s)",
                               (car_count_lane1, car_count_lane2, signal_status, route_recommendation))
                db.commit()
            except Exception as e:
                print(f"Database error: {e}")
                db.rollback()

            # Evaluate model every 10 frames
            if frame_count % 10 == 0 and frame_count > 0:
                # For demonstration, we'll use a simplified evaluation
                # In a real scenario, you would compare with actual ground truth
                try:
                    if os.path.exists("ground_truth.json"):
                        with open("ground_truth.json", "r") as f:
                            ground_truth_boxes = json.load(f)
                    else:
                        # Create dummy ground truth data if file doesn't exist
                        ground_truth_boxes = detected_boxes
                    
                    # Evaluate detection
                    if ground_truth_boxes and all_detected_boxes:
                        global_precision, global_recall, global_f1 = evaluate_model(all_detected_boxes, ground_truth_boxes)
                except Exception as e:
                    print(f"Evaluation error: {e}")

            # Add frame count to the image
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Cars detected: {car_count_lane1}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            frame_count += 1

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        camera.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route: Home Page
@app.route('/', methods=['GET', 'POST'])
def home():
    global global_precision, global_recall, global_f1
    
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No video file uploaded.", 400

        video = request.files['video']
        if video.filename == '':
            return "No selected file.", 400

        # Save the uploaded video
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4'))
        return redirect(url_for('home'))

    # Fetch the latest traffic data
    try:
        cursor.execute("SELECT * FROM traffic_data ORDER BY id DESC LIMIT 1")
        latest_data = cursor.fetchone()
        print("Latest Data from Database:", latest_data)  # Debugging
    except Exception as e:
        print(f"Database query error: {e}")
        latest_data = None

    traffic_data = {
        'lane1_cars': latest_data[2] if latest_data and len(latest_data) > 2 else 0,
        'lane2_cars': latest_data[3] if latest_data and len(latest_data) > 3 else 0,
        'signal_status': latest_data[4] if latest_data and len(latest_data) > 4 else "N/A",
        'route_recommendation': latest_data[5] if latest_data and len(latest_data) > 5 else "N/A"
    }

    return render_template('index.html', 
                           data=traffic_data, 
                           video_feed_url='/video_feed', 
                           precision=round(global_precision, 2), 
                           recall=round(global_recall, 2), 
                           f1=round(global_f1, 2))


# Helper function: Calculate IoU
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

# Helper function: Evaluate Model Performance
def evaluate_model(detected_boxes, ground_truth_boxes, iou_threshold=0.5):
    if not detected_boxes or not ground_truth_boxes:
        return 0, 0, 0
        
    y_true = []
    y_pred = []

    for gt_box in ground_truth_boxes:
        matched = any(calculate_iou(gt_box, det_box) >= iou_threshold for det_box in detected_boxes)
        y_true.append(1)
        y_pred.append(1 if matched else 0)

    for det_box in detected_boxes:
        matched = any(calculate_iou(gt_box, det_box) >= iou_threshold for gt_box in ground_truth_boxes)
        if not matched:
            y_true.append(0)
            y_pred.append(1)

    # Handle case where all predictions or ground truth are empty
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return 0, 0, 0

    try:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return 0, 0, 0

    return precision, recall, f1

@app.route('/evaluate_detection', methods=['POST'])
def evaluate_detection():
    global global_precision, global_recall, global_f1
    
    try:
        if os.path.exists("ground_truth.json"):
            with open("ground_truth.json", "r") as f:
                ground_truth_data = json.load(f)
        else:
            return jsonify({'error': 'Ground truth data not found'})

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')
        if not os.path.exists(video_path):
            video_path = 'traffic_video.mp4'
            
        camera = cv2.VideoCapture(video_path)

        if not camera.isOpened():
            return jsonify({'error': 'Could not open video file'})

        detected_boxes = []
        frame_count = 0

        while True:
            success, frame = camera.read()
            if not success:
                break

            _, frame_boxes, _ = detect_cars(frame)  # Extract detected bounding boxes
            detected_boxes.extend(frame_boxes)
            frame_count += 1

        camera.release()

        # Match detected boxes with ground truth
        global_precision, global_recall, global_f1 = evaluate_model(detected_boxes, ground_truth_data)

        return jsonify({
            'precision': round(global_precision, 2), 
            'recall': round(global_recall, 2), 
            'f1_score': round(global_f1, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)