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
from gemini_config import OBJECT_TO_TRACK, goal_setter_system_prompt

def get_initial_bounding_box(frame, object_to_track):
    try:
        # Use the improved goal_setter_system_prompt from config
        prompt = goal_setter_system_prompt.format(object=object_to_track)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS) 

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[prompt, pil_image],
        )

        if not response.text:
            print("Error: Received an empty response from the API.")
            return None

        print(f"DEBUG: API response text: {response.text}")
        
        # Try to extract JSON from the response
        response_text = response.text.strip()
        
        # Look for JSON in the response
        try:
            # Try to parse the whole response as JSON
            response_data = json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to find JSON within the text
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                print(f"Could not extract JSON from response: {response_text}")
                return "not_found"

        if isinstance(response_data, list) and response_data:
            response_data = response_data[0]

        if not isinstance(response_data, dict):
            print(f"Error: API response is not a dictionary: {response_data}")
            return None

        if "box_2d" not in response_data or response_data["box_2d"] is None:
            print(f"Warning: 'box_2d' not in response or is null. Object likely not found.")
            return "not_found"
            
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
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    camera.release()

def main():
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
    object_to_track = OBJECT_TO_TRACK
    capture_mode = False
    capture_start_time = 0
    capture_duration = 0.5 
    frame_for_detection = None

    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()

        if detection_future:
            if detection_future.done():
                result = detection_future.result()
                if isinstance(result, tuple): 
                    print(f"Initializing tracker with bbox: {result}")
                    
                    # Validate and convert bounding box format for CSRT
                    x, y, w, h = result
                    
                    # Ensure positive dimensions and within frame bounds
                    frame_h, frame_w = frame_for_detection.shape[:2]
                    x = max(0, min(x, frame_w - 1))
                    y = max(0, min(y, frame_h - 1))
                    w = max(1, min(w, frame_w - x))
                    h = max(1, min(h, frame_h - y))
                    
                    bbox_rect = (x, y, w, h)
                    print(f"Validated bbox: {bbox_rect} for frame size: {frame_w}x{frame_h}")
                    
                    # Use OpenCV CSRT tracker
                    tracker = cv2.TrackerCSRT_create()
                    success = tracker.init(frame_for_detection, bbox_rect)
                    
                    if success:
                        bbox = bbox_rect
                        print("CSRT tracker initialized successfully")
                    else:
                        print("Failed to initialize CSRT tracker")
                        # Try with legacy tracker as fallback
                        try:
                            tracker = cv2.legacy.TrackerCSRT_create()
                            success = tracker.init(frame_for_detection, bbox_rect)
                            if success:
                                bbox = bbox_rect
                                print("Legacy CSRT tracker initialized successfully")
                            else:
                                tracker = None
                        except:
                            print("Legacy tracker also failed")
                            tracker = None
                    detection_future = None
                    frame_for_detection = None
                elif result == "not_found":
                    print("Object not found in detection")
                    detection_future = None 
                    frame_for_detection = None 
            else:
                cv2.putText(frame, "Detecting...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        elif capture_mode:
            cv2.putText(frame, "Get Ready...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            if time.time() - capture_start_time > capture_duration:
                capture_mode = False
                frame_for_detection = frame.copy()
                detection_future = executor.submit(get_initial_bounding_box, frame_for_detection, object_to_track)

        elif tracker is None:
            if not detection_future: 
                 cv2.putText(frame, f"Yosua Hares|5025221270", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            success, bbox = tracker.update(frame)
            print(f"Tracker update - Success: {success}, BBox: {bbox}")

            if success:
                center_x = int(bbox[0] + bbox[2] / 2)
                center_y = int(bbox[1] + bbox[3] / 2)
                print(f"Drawing circle at: ({center_x}, {center_y})")
                
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                #              (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 0, 0), 2)
                
                cv2.circle(frame, (center_x, center_y), 8, (0, 255, 0), -1)
            else:
                cv2.putText(frame, "Tracking lost.", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                tracker = None


        cv2.imshow('Object Tracker', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and tracker is None and detection_future is None and not capture_mode:
            print("'s' || positive")
            capture_mode = True
            capture_start_time = time.time()
        elif key == ord('n') and detection_future is None and not capture_mode:
            print("'n' || positive")
            tracker = None
            bbox = None
            cv2.destroyAllWindows() 
            new_object = input("Enter the new object to track: ")
            if new_object:
                object_to_track = new_object
            cv2.imshow('Object Tracker', frame)


    executor.shutdown(wait=False)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
