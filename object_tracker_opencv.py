import os
import json
import time
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.5-pro"
OBJECT_TO_TRACK = "human nose"
PROMPT = f"""
  Pinpoint to the {OBJECT_TO_TRACK} in the image.
  The answer should be a single bounding box in the format: [y_min, x_min, y_max, x_max].
  The points are normalized to 0-1000.
"""

# --- Helper Functions ---
def get_initial_bounding_box(frame):
    """
    Uses the GenAI API to get the initial bounding box for the object.
    """
    print("Detecting object with GenAI...")
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image.thumbnail((800, 800), Image.Resampling.LANCZOS)

        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[pil_image, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        if not response.text:
            print("Error: Received an empty response from the API.")
            return None

        print(f"DEBUG: API response text: {response.text}")
        response_data = json.loads(response.text)

        if isinstance(response_data, list) and response_data:
            response_data = response_data[0]

        if not isinstance(response_data, dict):
            print(f"Error: API response is not a dictionary: {response_data}")
            return None

        if "box_2d" not in response_data:
            print(f"Error: 'box_2d' not in response: {response_data}")
            return None
            
        bbox_normalized = response_data['box_2d']
        if not isinstance(bbox_normalized, list) or len(bbox_normalized) != 4:
            print(f"Error: Invalid bounding box format: {bbox_normalized}")
            return None

        h, w, _ = frame.shape
        bbox = (
            int(bbox_normalized[1] / 1000 * w),
            int(bbox_normalized[0] / 1000 * h),
            int((bbox_normalized[3] - bbox_normalized[1]) / 1000 * w),
            int((bbox_normalized[2] - bbox_normalized[0]) / 1000 * h)
        )
        print(f"Object detected at: {bbox}")
        return bbox

    except Exception as e:
        print(f"Error during object detection: {e}")
        return None

def camera_thread(camera, frame_queue):
    """
    Continuously captures frames from the camera and puts them in a queue.
    """
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    camera.release()

def main():
    """
    Main function to run the object tracker.
    """
    if not API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_queue = Queue(maxsize=2)
    cam_thread = threading.Thread(target=camera_thread, args=(camera, frame_queue), daemon=True)
    cam_thread.start()

    executor = ThreadPoolExecutor(max_workers=1)
    detection_future = None

    tracker = None
    bbox = None

    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()

        if detection_future:
            if detection_future.done():
                initial_bbox = detection_future.result()
                if initial_bbox:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, initial_bbox)
                    bbox = initial_bbox
                detection_future = None
            else:
                cv2.putText(frame, "Detecting...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        elif tracker is None:
            cv2.putText(frame, "Press 's' to start tracking", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            success, bbox = tracker.update(frame)

            if success:
                center_x = int(bbox[0] + bbox[2] / 2)
                center_y = int(bbox[1] + bbox[3] / 2)
                cv2.circle(frame, (center_x, center_y), 8, (0, 255, 0), -1)
                cv2.putText(frame, "Tracking", (center_x - 30, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking failed! Re-acquiring...", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                tracker = None # Reset tracker
                if detection_future is None:
                    detection_future = executor.submit(get_initial_bounding_box, frame.copy())


        cv2.imshow('Object Tracker', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and tracker is None and detection_future is None:
            detection_future = executor.submit(get_initial_bounding_box, frame.copy())

    executor.shutdown(wait=False)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
