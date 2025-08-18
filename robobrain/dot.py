# format metadata
# {
#         "id": 0,
#         "image_path": "rtx_frames_success_0/10_utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds#episode_2/frame_0.png",
#         "meta_data": {
#             "original_dataset": "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
#             "original_width": 128,
#             "original_height": 128
#         },
#         "instruction": "reach for the cloth",
#         "trajectory": [
#             [
#                 100.33082706766918,
#                 39.93984962406015
#             ],
#             [
#                 88.06015037593984,
#                 44.03007518796992
#             ],
#             [
#                 73.62406015037594,
#                 44.99248120300752
#             ],
#             [
#                 63.5187969924812,
#                 40.902255639097746
#             ]
#         ]
#     },

import cv2
import numpy as np
import json
import os

IMAGE_PATH = '/Users/hares/Desktop/Magang BRIN/prototype/dataset/1e689e6992da4af78f6f250117ce4353.jpg'
WINDOW_NAME = 'Trajectory Annotation'

INSTRUCTION = "reach for the cloth"
METADATA = {
    "original_dataset": "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "original_width": 0,
    "original_height": 0
}

trajectory_points = []
current_color_name = 'x' 
colors = {
    'x': (0, 0, 255),  
    'y': (0, 255, 0),  
    'z': (255, 0, 0)   
}
point_colors = []
image = None
image_copy = None


def redraw_points():
    global image_copy
    if image is None:
        return
    image_copy = image.copy()
    for i, point in enumerate(trajectory_points):
        color = point_colors[i]
        cv2.circle(image_copy, tuple(point), 5, color, -1)
        if i > 0:
            prev_point = trajectory_points[i-1]
            cv2.line(image_copy, tuple(prev_point), tuple(point), color, 2)
    cv2.imshow(WINDOW_NAME, image_copy)

def handle_mouse_events(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        trajectory_points.append([x, y])
        point_colors.append(colors[current_color_name])
        print(f"Added point: ({x}, {y}) with color '{current_color_name}'")
        redraw_points()

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at '{IMAGE_PATH}'")
        print("Please update the IMAGE_PATH variable in the script.")
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        METADATA['original_width'] = 512
        METADATA['original_height'] = 512
    else:
        image = cv2.imread(IMAGE_PATH)
        if image is None:
            print(f"Error: Failed to load image from '{IMAGE_PATH}'. It might be corrupted or in an unsupported format.")
            exit()
        METADATA['original_width'] = image.shape[1]
        METADATA['original_height'] = image.shape[0]

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse_events)

    redraw_points()

    print("--- Trajectory Annotation Tool ---")
    print(f"Instruction: {INSTRUCTION}")
    print("\nControls:")
    print("  - Click on the image to add a trajectory point.")
    print("  - Press 'x' to select RED color.")
    print("  - Press 'y' to select GREEN color.")
    print("  - Press 'z' to select BLUE color.")
    print("  - Press 'u' to UNDO the last point.")
    print("  - Press 'c' to CLEAR all points.")
    print("  - Press 'q' to QUIT and generate the dataset.")
    print("---------------------------------")
    print(f"Current color: RED ('x')")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('x'):
            current_color_name = 'x'
            print("Current color set to RED ('x')")
        elif key == ord('y'):
            current_color_name = 'y'
            print("Current color set to GREEN ('y')")
        elif key == ord('z'):
            current_color_name = 'z'
            print("Current color set to BLUE ('z')")
        elif key == ord('u'): 
            if trajectory_points:
                trajectory_points.pop()
                point_colors.pop()
                redraw_points()
                print("Last point undone.")
            else:
                print("No points to undo.")
        elif key == ord('c'): 
            trajectory_points = []
            point_colors = []
            redraw_points()
            print("All points have been cleared.")

    output_data = {
        "id": 0,
        "image_path": IMAGE_PATH,
        "meta_data": METADATA,
        "instruction": INSTRUCTION,
        "trajectory": trajectory_points
    }

    print("\n--- Generated Trajectory Dataset ---")
    print(json.dumps(output_data, indent=4))
    print("------------------------------------")

    cv2.destroyAllWindows()
