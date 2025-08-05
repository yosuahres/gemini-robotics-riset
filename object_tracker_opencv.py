import os
import json
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
from scipy.spatial.distance import cdist
from collections import defaultdict, deque

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.5-pro"

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
        current_time = time.time()
        dt = current_time - self.last_update
        
        if len(self.position_history) > 0 and dt > 0:
            old_pos = self.position_history[-1]
            new_velocity = [
                (new_position[0] - old_pos[0]) / dt,
                (new_position[1] - old_pos[1]) / dt
            ]
            
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
        self.update_prediction()

    def update_prediction(self):
        current_time = time.time()
        dt = current_time - self.last_update
        self.prediction = [
            self.position[0] + self.velocity[0] * dt + 0.5 * self.acceleration[0] * dt * dt,
            self.position[1] + self.velocity[1] * dt + 0.5 * self.acceleration[1] * dt * dt
        ]
        self.interpolated_position = self.prediction.copy()

    def get_interpolated_position(self):
        current_time = time.time()
        dt = current_time - self.last_update
        
        if dt < 0.1:
            interpolated = [
                self.position[0] + self.velocity[0] * dt + 0.5 * self.acceleration[0] * dt * dt,
                self.position[1] + self.velocity[1] * dt + 0.5 * self.acceleration[1] * dt * dt
            ]
            damping_factor = max(0.1, 1.0 - dt * 2)
            interpolated = [
                self.position[0] + (interpolated[0] - self.position[0]) * damping_factor,
                self.position[1] + (interpolated[1] - self.position[1]) * damping_factor
            ]
            return interpolated
        else:
            return self.position

    def is_stale(self, max_age=5.0):
        return time.time() - self.last_seen > max_age

class ObjectTracker:
    def __init__(self, max_distance=100, max_age=5.0):
        self.tracked_objects = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_age = max_age

    def update(self, detected_points):
        if not detected_points:
            self._cleanup_stale_objects()
            return list(self.tracked_objects.values())
        
        detected_positions = np.array([point['point'] for point in detected_points])
        detected_labels = [point['label'] for point in detected_points]
        
        if self.tracked_objects:
            tracked_positions = np.array([obj.position for obj in self.tracked_objects.values()])
            tracked_ids = list(self.tracked_objects.keys())
            distance_matrix = cdist(detected_positions, tracked_positions)
            
            matched_detections = set()
            matched_tracks = set()
            
            for detection_idx in range(len(detected_points)):
                if detection_idx in matched_detections:
                    continue
                
                best_track_idx = None
                best_distance = float('inf')
                
                for track_idx in range(len(tracked_ids)):
                    if track_idx in matched_tracks:
                        continue
                    
                    distance = distance_matrix[detection_idx][track_idx]
                    track_id = tracked_ids[track_idx]
                    if (distance < self.max_distance and 
                        distance < best_distance and
                        self.tracked_objects[track_id].label == detected_labels[detection_idx]):
                        best_distance = distance
                        best_track_idx = track_idx
                
                if best_track_idx is not None:
                    track_id = tracked_ids[best_track_idx]
                    self.tracked_objects[track_id].update_position(detected_positions[detection_idx].tolist())
                    matched_detections.add(detection_idx)
                    matched_tracks.add(best_track_idx)
            
            for detection_idx in range(len(detected_points)):
                if detection_idx not in matched_detections:
                    new_id = f"obj_{self.next_id}"
                    self.next_id += 1
                    self.tracked_objects[new_id] = TrackedObject(new_id, detected_positions[detection_idx].tolist(), detected_labels[detection_idx])
        else:
            for i, point in enumerate(detected_points):
                new_id = f"obj_{self.next_id}"
                self.next_id += 1
                self.tracked_objects[new_id] = TrackedObject(new_id, point['point'], point['label'])
        
        self._cleanup_stale_objects()
        return list(self.tracked_objects.values())

    def _cleanup_stale_objects(self):
        stale_ids = [obj_id for obj_id, obj in self.tracked_objects.items() if obj.is_stale(self.max_age)]
        for obj_id in stale_ids:
            del self.tracked_objects[obj_id]

class CameraProcessor:
    def __init__(self, api_key, model_id, tracker):
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.tracker = tracker

    def process_frame(self, frame, prompt):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
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
            tracked_objects = self.tracker.update(detected_points)
            
            return tracked_objects
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return []

def main():
    if not API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    object_to_track = "human nose"
    prompt = f"""
      Pinpoint the {object_to_track} in the image.
      The answer should follow the json format: [{{"point": <point>, "label": "{object_to_track}"}}, ...]. 
      The points are in [y, x] format normalized to 0-1000. One element a line.
    """

    object_tracker = ObjectTracker(max_distance=150, max_age=3.0)
    processor = CameraProcessor(API_KEY, MODEL_ID, object_tracker)
    
    executor = ThreadPoolExecutor(max_workers=1)
    future = None
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        if future is None or future.done():
            future = executor.submit(processor.process_frame, frame.copy(), prompt)

        tracked_objects = list(object_tracker.tracked_objects.values())

        for obj in tracked_objects:
            pos = obj.get_interpolated_position()
            x = int(pos[1] / 1000.0 * frame.shape[1])
            y = int(pos[0] / 1000.0 * frame.shape[0])
            
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"{obj.label} ({obj.id})", (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Object Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    executor.shutdown()
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
