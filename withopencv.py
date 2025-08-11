import os
import json
import time
import cv2
import numpy as np
from PIL import Image
from google import genai
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from gemini_config import scene_analyzer_prompt, verification_system_prompt

def analyze_scene(frame):
    """
    Analyzes the entire scene to detect all relevant objects and their functions.
    """
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[scene_analyzer_prompt, pil_image],
        )

        if not response.text:
            print("Error: Received an empty response from the API during scene analysis.")
            return None

        print(f"DEBUG: Scene analysis API response text: {response.text}")

        response_text = response.text.strip()
        
        try:
            # Handle potential markdown code block
            if response_text.startswith("```json"):
                response_text = response_text[7:-4].strip()
            response_data = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                print(f"Could not extract JSON from scene analysis response: {response_text}")
                return None

        if isinstance(response_data, dict) and "objects" in response_data:
            return response_data["objects"]
        else:
            print(f"Error: API response is not in the expected format: {response_data}")
            return None

    except Exception as e:
        print(f"Error during scene analysis: {e}")
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
    analysis_future = None
    tracker = None
    bbox = None
    scene_state = None # This will store our object map
    frame_for_analysis = None

    # Listen for terminal input in a separate thread to not block the CV window
    def terminal_input_thread():
        nonlocal tracker, scene_state, bbox, frame_for_analysis
        while True:
            try:
                object_to_track = input("Enter object to track: ").lower().strip()
                if not object_to_track:
                    continue

                if scene_state and object_to_track in scene_state:
                    print(f"Attempting to track '{object_to_track}'...")
                    obj_data = scene_state[object_to_track]
                    bbox_normalized = obj_data['box_2d']
                    
                    if frame_for_analysis is None:
                        print("Error: Reference frame for analysis is missing.")
                        continue

                    h, w, _ = frame_for_analysis.shape
                    bbox_rect = (
                        int(bbox_normalized[1] / 1000 * w),
                        int(bbox_normalized[0] / 1000 * h),
                        int((bbox_normalized[3] - bbox_normalized[1]) / 1000 * w),
                        int((bbox_normalized[2] - bbox_normalized[0]) / 1000 * h)
                    )

                    x, y, w_box, h_box = bbox_rect
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    w_box = max(1, min(w_box, w - x))
                    h_box = max(1, min(h_box, h - y))
                    bbox_rect = (x, y, w_box, h_box)

                    new_tracker = cv2.TrackerCSRT_create()
                    success = new_tracker.init(frame_for_analysis, bbox_rect) 
                    
                    if success:
                        tracker = new_tracker
                        bbox = bbox_rect
                        print(f"Successfully initialized tracker for '{object_to_track}'.")
                    else:
                        print(f"Failed to initialize tracker for '{object_to_track}'.")
                        tracker = None
                elif scene_state:
                    print(f"Object '{object_to_track}' not found. Available objects: {list(scene_state.keys())}")
                else:
                    print("Please scan the scene first by pressing 's'.")

            except (EOFError, KeyboardInterrupt):
                print("\nInput thread stopped.")
                break
            except Exception as e:
                print(f"Error in input thread: {e}")
                break

    input_thread = threading.Thread(target=terminal_input_thread, daemon=True)
    input_thread.start()

    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()

        if analysis_future:
            if analysis_future.done():
                result = analysis_future.result()
                if result:
                    print("\n--- Full JSON Response ---")
                    print(json.dumps(result, indent=2))
                    print("--------------------------\n")
                    scene_state = {obj['object_name'].lower(): obj for obj in result}
                    print("Scene analysis complete. Detected objects:")
                    for name, obj_data in scene_state.items():
                        print(f"- {name}: {obj_data['function']}")
                    print("\nEnter the name of the object to track in the terminal.")
                else:
                    print("Scene analysis failed.")
                analysis_future = None
            else:
                cv2.putText(frame, "Analyzing Scene...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        elif scene_state and tracker is None:
             # Draw boxes for all detected objects if we have a scene state but aren't tracking yet
            h, w, _ = frame.shape
            for name, obj_data in scene_state.items():
                bbox_normalized = obj_data['box_2d']
                box = (
                    int(bbox_normalized[1] / 1000 * w),
                    int(bbox_normalized[0] / 1000 * h),
                    int((bbox_normalized[3] - bbox_normalized[1]) / 1000 * w),
                    int((bbox_normalized[2] - bbox_normalized[0]) / 1000 * h)
                )
                p1 = (box[0], box[1])
                p2 = (box[0] + box[2], box[1] + box[3])
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
                cv2.putText(frame, name, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Enter object name in terminal to track", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)


        elif tracker:
            success, bbox = tracker.update(frame)
            if success:
                center_x = int(bbox[0] + bbox[2] / 2)
                center_y = int(bbox[1] + bbox[3] / 2)
                cv2.circle(frame, (center_x, center_y), 8, (0, 255, 0), -1)
            else:
                cv2.putText(frame, "Tracking lost", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                tracker = None # Reset tracker on failure

        else: # Default state, waiting for action
            cv2.putText(frame, "Press 's' to scan scene", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


        cv2.imshow('Object Tracker', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not analysis_future and not scene_state:
            print("Starting scene analysis...")
            frame_for_analysis = frame.copy()
            analysis_future = executor.submit(analyze_scene, frame_for_analysis)
        elif key == ord('n'): # Reset key
            print("Resetting state.")
            tracker = None
            bbox = None
            scene_state = None
            analysis_future = None

    executor.shutdown(wait=True)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
