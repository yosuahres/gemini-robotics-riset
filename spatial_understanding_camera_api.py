import os
import json
import base64
import threading
import time
import uuid
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import cv2
import numpy as np
from PIL import Image, ImageDraw
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from scipy.spatial.distance import cdist
from collections import defaultdict, deque

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.5-pro"

app = FastAPI(title="Spatial Understanding Camera API")

# Global variables for camera and processing
camera = None
latest_frame = None
latest_points = []
tracked_objects = {}  # Dictionary to store tracked objects
processing_queue = Queue()
is_processing = False
frame_lock = threading.Lock()
object_tracker = None
custom_prompt = None

class TrackedObject:
    def __init__(self, obj_id, position, label, confidence=1.0):
        self.id = obj_id
        self.label = label
        self.position = position  # [y, x] format
        self.confidence = confidence
        self.last_seen = time.time()
        self.last_update = time.time()
        self.velocity = [0.0, 0.0]  # [dy, dx] per frame
        self.acceleration = [0.0, 0.0]  # [d²y, d²x] per frame²
        self.position_history = deque(maxlen=10)
        self.position_history.append(position)
        self.prediction = position.copy()
        self.interpolated_position = position.copy()
        
    def update_position(self, new_position, confidence=1.0):
        """Update object position and calculate velocity and acceleration"""
        current_time = time.time()
        dt = current_time - self.last_update
        
        if len(self.position_history) > 0 and dt > 0:
            old_pos = self.position_history[-1]
            new_velocity = [
                (new_position[0] - old_pos[0]) / dt,
                (new_position[1] - old_pos[1]) / dt
            ]
            
            # Calculate acceleration
            if dt > 0:
                self.acceleration = [
                    (new_velocity[0] - self.velocity[0]) / dt,
                    (new_velocity[1] - self.velocity[1]) / dt
                ]
            
            self.velocity = new_velocity
        
        self.position = new_position
        self.interpolated_position = new_position.copy()
        self.confidence = confidence
        self.last_seen = current_time
        self.last_update = current_time
        self.position_history.append(new_position)
        
        # Update prediction based on velocity and acceleration
        self.update_prediction()
    
    def update_prediction(self):
        """Update prediction based on current velocity and acceleration"""
        current_time = time.time()
        dt = current_time - self.last_update
        
        # Use kinematic equations: position = initial_pos + velocity*time + 0.5*acceleration*time²
        self.prediction = [
            self.position[0] + self.velocity[0] * dt + 0.5 * self.acceleration[0] * dt * dt,
            self.position[1] + self.velocity[1] * dt + 0.5 * self.acceleration[1] * dt * dt
        ]
        
        # Update interpolated position for smooth movement
        self.interpolated_position = self.prediction.copy()
    
    def get_interpolated_position(self):
        """Get current interpolated position based on time since last update"""
        current_time = time.time()
        dt = current_time - self.last_update
        
        if dt < 0.1:  # Only interpolate for short time periods
            # Use kinematic equations for smooth interpolation
            interpolated = [
                self.position[0] + self.velocity[0] * dt + 0.5 * self.acceleration[0] * dt * dt,
                self.position[1] + self.velocity[1] * dt + 0.5 * self.acceleration[1] * dt * dt
            ]
            
            # Apply some damping to prevent wild extrapolation
            damping_factor = max(0.1, 1.0 - dt * 2)  # Reduce confidence over time
            interpolated = [
                self.position[0] + (interpolated[0] - self.position[0]) * damping_factor,
                self.position[1] + (interpolated[1] - self.position[1]) * damping_factor
            ]
            
            return interpolated
        else:
            # For longer periods, just return the last known position
            return self.position
    
    def predict_next_position(self):
        """Predict next position based on velocity and acceleration"""
        return self.get_interpolated_position()
    
    def is_stale(self, max_age=5.0):
        """Check if object hasn't been seen for too long"""
        return time.time() - self.last_seen > max_age

class ObjectTracker:
    def __init__(self, max_distance=100, max_age=5.0):
        self.tracked_objects = {}
        self.next_id = 1
        self.max_distance = max_distance  # Maximum distance for object matching
        self.max_age = max_age  # Maximum age before object is considered lost
        
    def update(self, detected_points):
        """Update tracker with new detected points"""
        if not detected_points:
            # Just update predictions for existing objects
            self._cleanup_stale_objects()
            return list(self.tracked_objects.values())
        
        # Convert detected points to numpy array for distance calculation
        detected_positions = np.array([point['point'] for point in detected_points])
        detected_labels = [point['label'] for point in detected_points]
        
        # Get current tracked positions
        if self.tracked_objects:
            tracked_positions = np.array([obj.position for obj in self.tracked_objects.values()])
            tracked_ids = list(self.tracked_objects.keys())
            
            # Calculate distance matrix
            distance_matrix = cdist(detected_positions, tracked_positions)
            
            # Hungarian algorithm would be ideal here, but for simplicity, use greedy matching
            matched_detections = set()
            matched_tracks = set()
            
            # Match detections to existing tracks
            for detection_idx in range(len(detected_points)):
                if detection_idx in matched_detections:
                    continue
                    
                best_track_idx = None
                best_distance = float('inf')
                
                for track_idx in range(len(tracked_ids)):
                    if track_idx in matched_tracks:
                        continue
                    
                    distance = distance_matrix[detection_idx][track_idx]
                    
                    # Check if labels match and distance is reasonable
                    track_id = tracked_ids[track_idx]
                    if (distance < self.max_distance and 
                        distance < best_distance and
                        self.tracked_objects[track_id].label == detected_labels[detection_idx]):
                        best_distance = distance
                        best_track_idx = track_idx
                
                # Update matched track
                if best_track_idx is not None:
                    track_id = tracked_ids[best_track_idx]
                    self.tracked_objects[track_id].update_position(
                        detected_positions[detection_idx].tolist()
                    )
                    matched_detections.add(detection_idx)
                    matched_tracks.add(best_track_idx)
            
            # Create new tracks for unmatched detections
            for detection_idx in range(len(detected_points)):
                if detection_idx not in matched_detections:
                    new_id = f"obj_{self.next_id}"
                    self.next_id += 1
                    
                    self.tracked_objects[new_id] = TrackedObject(
                        new_id,
                        detected_positions[detection_idx].tolist(),
                        detected_labels[detection_idx]
                    )
        else:
            # No existing tracks, create new ones for all detections
            for i, point in enumerate(detected_points):
                new_id = f"obj_{self.next_id}"
                self.next_id += 1
                
                self.tracked_objects[new_id] = TrackedObject(
                    new_id,
                    point['point'],
                    point['label']
                )
        
        # Clean up stale objects
        self._cleanup_stale_objects()
        
        return list(self.tracked_objects.values())
    
    def _cleanup_stale_objects(self):
        """Remove objects that haven't been seen for too long"""
        stale_ids = [
            obj_id for obj_id, obj in self.tracked_objects.items()
            if obj.is_stale(self.max_age)
        ]
        
        for obj_id in stale_ids:
            del self.tracked_objects[obj_id]
    
    def get_all_objects(self):
        """Get all currently tracked objects"""
        return list(self.tracked_objects.values())
    
    def get_interpolated_objects(self):
        """Get all tracked objects with interpolated positions for smooth movement"""
        interpolated_objects = []
        for obj in self.tracked_objects.values():
            # Create a copy with interpolated position
            interpolated_obj = type('InterpolatedObject', (), {
                'id': obj.id,
                'label': obj.label,
                'position': obj.get_interpolated_position(),
                'confidence': obj.confidence,
                'velocity': obj.velocity,
                'acceleration': obj.acceleration,
                'prediction': obj.predict_next_position(),
                'last_seen': obj.last_seen
            })()
            interpolated_objects.append(interpolated_obj)
        return interpolated_objects

class CameraProcessor:
    def __init__(self, api_key, model_id, tracker):
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.tracker = tracker
        
    def process_frame(self, frame, prompt=None):
        """Process a frame and return tracked points data"""
        try:
            # Convert OpenCV frame to PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Resize for processing
            pil_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            if prompt is None:
                prompt = """
                  Pinpoint to human nose, hands, and any objects they are interacting with.
                  The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. 
                  The points are in [y, x] format normalized to 0-1000. One element a line.
                  Be specific with labels (e.g., "person_nose", "left_hand", "right_hand", "cup", "phone", etc.)
                """
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[pil_image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            if not response.text:
                print("Error: Received an empty response from the API.")
                return []
                
            detected_points = json.loads(response.text)
            
            # Update tracker with detected points
            tracked_objects = self.tracker.update(detected_points)
            
            # Convert tracked objects back to points format for API compatibility
            tracked_points = []
            for obj in tracked_objects:
                tracked_points.append({
                    "point": obj.position,
                    "label": obj.label,
                    "id": obj.id,
                    "confidence": obj.confidence,
                    "velocity": obj.velocity,
                    "prediction": obj.prediction
                })
            
            return tracked_points
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return []

# Initialize camera and processor
def initialize_camera():
    global camera
    try:
        camera = cv2.VideoCapture(0)  # Use default camera
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

def camera_thread():
    """Continuously capture frames from camera"""
    global latest_frame, camera
    
    while camera and camera.isOpened():
        ret, frame = camera.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        time.sleep(0.033)  # ~30 FPS

def processing_thread():
    """Process frames in background with object tracking"""
    global latest_points, is_processing, object_tracker
    
    # Initialize tracker
    object_tracker = ObjectTracker(max_distance=50, max_age=3.0)
    processor = CameraProcessor(API_KEY, MODEL_ID, object_tracker)
    
    while True:
        if not processing_queue.empty():
            frame, prompt = processing_queue.get()
            is_processing = True
            
            # Process frame in executor for non-blocking operation
            future = processor.executor.submit(processor.process_frame, frame, prompt)
            try:
                points = future.result(timeout=30)  # 30 second timeout
                latest_points = points
                print(f"Processed frame, tracking {len(points)} objects")
            except Exception as e:
                print(f"Processing error: {e}")
                latest_points = []
            finally:
                is_processing = False
        
        # Continuously update object predictions for smooth tracking
        if object_tracker:
            tracked_objects = object_tracker.get_interpolated_objects()
            if tracked_objects:
                # Update positions with interpolated positions for smooth tracking
                interpolated_points = []
                for obj in tracked_objects:
                    interpolated_points.append({
                        "point": obj.position,  # Use interpolated position for smooth movement
                        "label": obj.label,
                        "id": obj.id,
                        "confidence": obj.confidence,
                        "velocity": obj.velocity,
                        "acceleration": getattr(obj, 'acceleration', [0, 0]),
                        "prediction": obj.prediction,
                        "last_seen": obj.last_seen
                    })
                
                # Only update if we have tracked objects or if processing just finished
                if interpolated_points and (not processing_queue.empty() or len(latest_points) == 0):
                    latest_points = interpolated_points
        
        time.sleep(0.033)  # ~30 FPS for smooth interpolation

@app.on_event("startup")
async def startup_event():
    """Initialize camera and start background threads"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not found in environment")
    
    if not initialize_camera():
        raise HTTPException(status_code=500, detail="Failed to initialize camera")
    
    # Start camera capture thread
    camera_thread_instance = threading.Thread(target=camera_thread, daemon=True)
    camera_thread_instance.start()
    
    # Start processing thread
    processing_thread_instance = threading.Thread(target=processing_thread, daemon=True)
    processing_thread_instance.start()
    
    print("Camera and processing threads started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup camera resources"""
    global camera
    if camera:
        camera.release()
        
@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main HTML page"""
    return generate_camera_html()

@app.get("/api/frame")
async def get_current_frame():
    """Get the current camera frame as base64"""
    global latest_frame
    
    if latest_frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    
    with frame_lock:
        frame = latest_frame.copy()
    
    # Convert to PIL and then to base64
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return JSONResponse({
        "image": img_str,
        "timestamp": time.time()
    })

@app.get("/api/points")
async def get_current_points():
    """Get the latest processed points"""
    return JSONResponse({
        "points": latest_points,
        "is_processing": is_processing,
        "timestamp": time.time()
    })

@app.get("/api/points/realtime")
async def get_realtime_points():
    """Get real-time interpolated points for smooth tracking"""
    global object_tracker
    
    if object_tracker:
        tracked_objects = object_tracker.get_interpolated_objects()
        realtime_points = []
        for obj in tracked_objects:
            realtime_points.append({
                "point": obj.position,
                "label": obj.label,
                "id": obj.id,
                "confidence": obj.confidence,
                "velocity": obj.velocity,
                "acceleration": getattr(obj, 'acceleration', [0, 0]),
                "prediction": obj.prediction,
                "last_seen": obj.last_seen
            })
        
        return JSONResponse({
            "points": realtime_points,
            "timestamp": time.time()
        })
    else:
        return JSONResponse({
            "points": [],
            "timestamp": time.time()
        })

@app.post("/api/process")
async def trigger_processing():
    """Trigger processing of the current frame"""
    global latest_frame, processing_queue, custom_prompt
    
    if latest_frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    
    if processing_queue.qsize() < 2:  # Limit queue size
        with frame_lock:
            processing_queue.put((latest_frame.copy(), custom_prompt))
        return JSONResponse({"status": "processing_queued"})
    else:
        return JSONResponse({"status": "queue_full"})

@app.post("/api/prompt")
async def set_prompt(prompt_data: dict):
    """Set the custom prompt for object detection"""
    global custom_prompt
    prompt = prompt_data.get("prompt")
    if prompt:
        custom_prompt = prompt
        return JSONResponse({"status": "prompt_updated"})
    else:
        raise HTTPException(status_code=400, detail="Prompt not provided")

def generate_camera_html():
    """Generate the HTML page for camera feed with live processing"""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Live Spatial Understanding</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}

        .camera-container {{
            position: relative;
            display: inline-block;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        #cameraFeed {{
            display: block;
            max-width: 800px;
            height: auto;
        }}

        .point-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}

        .point {{
            position: absolute;
            width: 12px;
            height: 12px;
            background-color: #2962FF;
            border: 2px solid #fff;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 0 40px rgba(41, 98, 255, 0.6);
            opacity: 1;
            transition: all 0.1s cubic-bezier(0.25, 0.8, 0.25, 1);
            pointer-events: auto;
            z-index: 10;
        }}

        .point.tracked {{
            background-color: #4CAF50;
            box-shadow: 0 0 40px rgba(76, 175, 80, 0.6);
        }}

        .point.new {{
            opacity: 0;
            animation: pointFadeIn 0.5s forwards;
        }}

        .point.moving {{
            background-color: #FF9800;
            box-shadow: 0 0 40px rgba(255, 152, 0, 0.8);
            animation: pulse 1s infinite;
        }}

        .point.realtime {{
            transition: all 0.05s linear;
        }}

        @keyframes pulse {{
            0% {{ transform: translate(-50%, -50%) scale(1); }}
            50% {{ transform: translate(-50%, -50%) scale(1.2); }}
            100% {{ transform: translate(-50%, -50%) scale(1); }}
        }}

        .point.new {{
            opacity: 0;
            animation: pointFadeIn 0.5s forwards;
        }}

        @keyframes pointFadeIn {{
            from {{
                opacity: 0;
                transform: translate(-50%, -50%) scale(0.5);
            }}
            to {{
                opacity: 1;
                transform: translate(-50%, -50%) scale(1);
            }}
        }}

        .point.fade-out {{
            animation: pointFadeOut 0.3s forwards;
        }}

        .point.highlight {{
            transform: translate(-50%, -50%) scale(1.1);
            background-color: #FF4081;
            box-shadow: 0 0 40px rgba(255, 64, 129, 0.6);
            z-index: 100;
        }}

        @keyframes pointFadeOut {{
            from {{
                opacity: 1;
            }}
            to {{
                opacity: 0.7;
            }}
        }}

        .point-label {{
            position: absolute;
            background-color: #2962FF;
            color: #fff;
            font-size: 11px;
            padding: 3px 6px;
            border-radius: 3px;
            transform: translate(20px, -10px);
            white-space: nowrap;
            opacity: 1;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            box-shadow: 0 0 30px rgba(41, 98, 255, 0.4);
            pointer-events: auto;
            cursor: pointer;
            z-index: 11;
        }}

        .point-label.tracked {{
            background-color: #4CAF50;
            box-shadow: 0 0 30px rgba(76, 175, 80, 0.4);
        }}

        .point-label.moving {{
            background-color: #FF9800;
            box-shadow: 0 0 30px rgba(255, 152, 0, 0.4);
        }}

        .object-id {{
            font-size: 9px;
            opacity: 0.8;
            display: block;
        }}

        .point-label.new {{
            opacity: 0;
        }}

        .point-label.fade-out {{
            opacity: 0.45;
        }}

        .point-label.highlight {{
            background-color: #FF4081;
            box-shadow: 0 0 30px rgba(255, 64, 129, 0.4);
            transform: translate(20px, -10px) scale(1.1);
            z-index: 100;
        }}

        .controls {{
            margin-top: 20px;
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .btn {{
            background: #2962FF;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            margin: 0 10px;
            transition: background 0.2s;
        }}

        .btn:hover {{
            background: #1e4ddf;
        }}

        .btn:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}

        .status {{
            margin-top: 10px;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 4px;
            display: inline-block;
        }}

        .processing {{
            background: #fff3e0;
        }}

        .info-panel {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Live Spatial Understanding</h1>
            <p>Real-time object detection and spatial analysis using camera feed</p>
        </div>

        <div class="camera-container">
            <img id="cameraFeed" src="" alt="Camera Feed">
            <div id="pointOverlay" class="point-overlay"></div>
        </div>

        <div class="controls">
            <input type="text" id="promptInput" placeholder="Enter a prompt to track an object..." style="padding: 12px; border-radius: 6px; border: 1px solid #ccc; font-size: 14px; min-width: 300px;">
            <button id="promptBtn" class="btn" onclick="setPrompt()">Set Prompt</button>
            <button id="processBtn" class="btn" onclick="processFrame()">Analyze Current Frame</button>
            <button id="autoBtn" class="btn" onclick="toggleAuto()">Start Auto Analysis</button>
            <div id="status" class="status">Ready</div>
        </div>

        <div class="info-panel">
            <h3>Detected Objects</h3>
            <div id="objectList">Click "Analyze Current Frame" to detect objects</div>
        </div>
    </div>

    <script>
        let autoMode = false;
        let autoInterval = null;
        let currentPoints = [];
        let customPrompt = null;
        let lastPointsHash = '';
        let existingPoints = new Map(); // Track existing points for smooth transitions

        function hashPoints(points) {{
            return JSON.stringify(points.map(p => [p.point, p.label, p.id || 'no-id']));
        }}

        function calculateDistance(pos1, pos2) {{
            const dy = pos1[0] - pos2[0];
            const dx = pos1[1] - pos2[1];
            return Math.sqrt(dy * dy + dx * dx);
        }}

        function isMoving(velocity) {{
            if (!velocity || velocity.length !== 2) return false;
            const speed = Math.sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1]);
            return speed > 10; // Threshold for considering an object as moving (increased for real-time)
        }}

        async function updateFrame() {{
            try {{
                const response = await fetch('/api/frame');
                const data = await response.json();
                const img = document.getElementById('cameraFeed');
                img.src = 'data:image/png;base64,' + data.image;
            }} catch (error) {{
                console.error('Error updating frame:', error);
            }}
        }}

        async function updatePointsRealtime() {{
            try {{
                const response = await fetch('/api/points/realtime');
                const data = await response.json();
                
                if (data.points && data.points.length > 0) {{
                    currentPoints = data.points;
                    renderPointsSmooth(data.points, true); // Pass true for realtime
                    updateObjectList(data.points);
                }}
            }} catch (error) {{
                console.error('Error updating real-time points:', error);
            }}
        }}

        async function updatePoints() {{
            try {{
                const response = await fetch('/api/points');
                const data = await response.json();
                
                updateStatus(data.is_processing);
                
                // Only update if we have new processed data or no real-time data
                if (data.points && data.points.length > 0) {{
                    const newHash = hashPoints(data.points);
                    if (newHash !== lastPointsHash) {{
                        currentPoints = data.points;
                        renderPointsSmooth(data.points, false); // Pass false for regular updates
                        updateObjectList(data.points);
                        lastPointsHash = newHash;
                    }}
                }}
            }} catch (error) {{
                console.error('Error updating points:', error);
            }}
        }}

        async function processFrame() {{
            try {{
                document.getElementById('processBtn').disabled = true;
                const response = await fetch('/api/process', {{ method: 'POST' }});
                const data = await response.json();
                
                if (data.status === 'processing_queued') {{
                    updateStatus(true);
                    // Poll for results
                    setTimeout(checkProcessingComplete, 1000);
                }}
            }} catch (error) {{
                console.error('Error processing frame:', error);
            }} finally {{
                setTimeout(() => {{
                    document.getElementById('processBtn').disabled = false;
                }}, 2000);
            }}
        }}

        async function setPrompt() {{
            const promptInput = document.getElementById('promptInput');
            const prompt = promptInput.value;
            if (!prompt) {{
                alert("Please enter a prompt.");
                return;
            }}
            
            customPrompt = `Pinpoint the ${{prompt}} in the image. The answer should follow the json format: [{{"point": <point>, "label": <label1>}}, ...]. The points are in [y, x] format normalized to 0-1000. One element a line.`;

            try {{
                const response = await fetch('/api/prompt', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{ prompt: customPrompt }})
                }});
                const data = await response.json();
                if (data.status === 'prompt_updated') {{
                    alert("Prompt updated successfully!");
                }}
            }} catch (error) {{
                console.error('Error setting prompt:', error);
            }}
        }}

        function checkProcessingComplete() {{
            updatePoints();
            // Check again if still processing
            fetch('/api/points')
                .then(response => response.json())
                .then(data => {{
                    if (data.is_processing) {{
                        setTimeout(checkProcessingComplete, 1000);
                    }}
                }});
        }}

        function toggleAuto() {{
            const btn = document.getElementById('autoBtn');
            if (autoMode) {{
                clearInterval(autoInterval);
                autoMode = false;
                btn.textContent = 'Start Auto Analysis';
                btn.style.background = '#2962FF';
            }} else {{
                autoInterval = setInterval(processFrame, 2000); // Process every 2 seconds for faster tracking
                autoMode = true;
                btn.textContent = 'Stop Auto Analysis';
                btn.style.background = '#FF4081';
            }}
        }}

        function renderPointsSmooth(points, isRealtime = false) {{
            const overlay = document.getElementById('pointOverlay');
            const newPointIds = new Set();

            points.forEach((pointData, index) => {{
                if (!pointData.hasOwnProperty("point")) return;

                const pointId = pointData.id || `point_${{index}}`;
                newPointIds.add(pointId);
                
                let point = document.getElementById(`point_${{pointId}}`);
                let pointLabel = point ? point.querySelector('.point-label') : null;
                
                const [y, x] = pointData.point;
                const newLeft = `${{x/1000.0 * 100.0}}%`;
                const newTop = `${{y/1000.0 * 100.0}}%`;
                
                if (point) {{
                    // Update existing point position smoothly
                    point.style.left = newLeft;
                    point.style.top = newTop;
                    
                    // Add realtime class for faster transitions
                    if (isRealtime) {{
                        point.classList.add('realtime');
                        pointLabel.classList.add('realtime');
                    }}
                    
                    // Update classes based on movement
                    point.classList.remove('moving', 'tracked');
                    pointLabel.classList.remove('moving', 'tracked');
                    
                    if (isMoving(pointData.velocity)) {{
                        point.classList.add('moving');
                        pointLabel.classList.add('moving');
                    }} else if (pointData.id) {{
                        point.classList.add('tracked');
                        pointLabel.classList.add('tracked');
                    }}
                    
                    // Update label content
                    const labelText = pointData.id ? 
                        `${{pointData.label}}<span class="object-id">${{pointData.id}}</span>` : 
                        pointData.label;
                    pointLabel.innerHTML = labelText;
                }} else {{
                    // Create new point
                    point = document.createElement('div');
                    point.className = isRealtime ? 'point realtime' : 'point new';
                    point.id = `point_${{pointId}}`;
                    point.style.left = newLeft;
                    point.style.top = newTop;

                    pointLabel = document.createElement('div');
                    pointLabel.className = isRealtime ? 'point-label realtime' : 'point-label new';
                    const labelText = pointData.id ? 
                        `${{pointData.label}}<span class="object-id">${{pointData.id}}</span>` : 
                        pointData.label;
                    pointLabel.innerHTML = labelText;
                    point.appendChild(pointLabel);

                    overlay.appendChild(point);

                    // Trigger animation after a small delay (only for new non-realtime points)
                    if (!isRealtime) {{
                        setTimeout(() => {{
                            point.classList.remove('new');
                            pointLabel.classList.remove('new');
                            
                            if (isMoving(pointData.velocity)) {{
                                point.classList.add('moving');
                                pointLabel.classList.add('moving');
                            }} else if (pointData.id) {{
                                point.classList.add('tracked');
                                pointLabel.classList.add('tracked');
                            }}
                        }}, index * 50 + 10);
                    }} else {{
                        // For realtime, immediately apply classes
                        if (isMoving(pointData.velocity)) {{
                            point.classList.add('moving');
                            pointLabel.classList.add('moving');
                        }} else if (pointData.id) {{
                            point.classList.add('tracked');
                            pointLabel.classList.add('tracked');
                        }}
                    }}

                    // Add hover effects
                    const handleMouseEnter = () => {{
                        point.classList.add('highlight');
                        pointLabel.classList.add('highlight');
                        
                        overlay.querySelectorAll('.point').forEach(p => {{
                            if (p !== point) {{
                                p.classList.add('fade-out');
                                p.querySelector('.point-label').classList.add('fade-out');
                            }}
                        }});
                    }};

                    const handleMouseLeave = () => {{
                        point.classList.remove('highlight');
                        pointLabel.classList.remove('highlight');
                        
                        overlay.querySelectorAll('.point').forEach(p => {{
                            p.classList.remove('fade-out');
                            p.querySelector('.point-label').classList.remove('fade-out');
                        }});
                    }};

                    point.addEventListener('mouseenter', handleMouseEnter);
                    point.addEventListener('mouseleave', handleMouseLeave);
                }}
            }});

            // Remove points that are no longer detected (only if not a realtime update)
            if (!isRealtime) {{
                overlay.querySelectorAll('.point').forEach(point => {{
                    const pointId = point.id.replace('point_', '');
                    if (!newPointIds.has(pointId)) {{
                        point.style.opacity = '0';
                        setTimeout(() => {{
                            if (point.parentNode) {{
                                point.parentNode.removeChild(point);
                            }}
                        }}, 300);
                    }}
                }});
            }}
        }}

        function updateObjectList(points) {{
            const list = document.getElementById('objectList');
            if (points.length === 0) {{
                list.innerHTML = 'No objects detected';
                return;
            }}

            const objectsByType = {{}};
            points.forEach(p => {{
                const type = p.label;
                if (!objectsByType[type]) {{
                    objectsByType[type] = [];
                }}
                objectsByType[type].push(p);
            }});

            let html = '<div>';
            Object.keys(objectsByType).forEach(type => {{
                const objects = objectsByType[type];
                html += `<div><strong>${{type}}:</strong> ${{objects.length}} tracked</div>`;
                objects.forEach(obj => {{
                    const status = isMoving(obj.velocity) ? '🔄 Moving' : '📍 Tracked';
                    html += `<div style="margin-left: 20px; font-size: 12px;">
                        ${{obj.id || 'Unknown'}} - ${{status}}
                    </div>`;
                }});
            }});
            html += '</div>';

            list.innerHTML = html;
        }}

        function updateStatus(isProcessing) {{
            const status = document.getElementById('status');
            if (isProcessing) {{
                status.textContent = 'Processing...';
                status.className = 'status processing';
            }} else {{
                const trackedCount = currentPoints.filter(p => p.id).length;
                const movingCount = currentPoints.filter(p => isMoving(p.velocity)).length;
                status.textContent = `Ready - ${{trackedCount}} objects tracked (${{movingCount}} moving)`;
                status.className = 'status';
            }}
        }}

        // Start frame updates
        setInterval(updateFrame, 100); // Update frame at ~10 FPS
        setInterval(updatePointsRealtime, 33); // Real-time updates at ~30 FPS for smooth tracking
        setInterval(updatePoints, 1000); // Check for new processed points every 1 second

        // Initial load
        updateFrame();
        updatePoints();
        updatePointsRealtime();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
