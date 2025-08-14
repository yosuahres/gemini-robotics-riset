# please refer to https://github.com/FlagOpen/RoboBrain
from inference import SimpleInference
from PIL import Image, ImageDraw
import ast

model_id = "BAAI/RoboBrain"
lora_id = "BAAI/RoboBrain-LoRA-Affordance"
model = SimpleInference(model_id, lora_id)
# Example 1:
prompt = "You are a robot using the joint control. The task is \"reach for the cloth\". Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point."
image_path = "./assets/demo/trajectory_1.jpg"
pred = model.inference(prompt, image_path, do_sample=False)
print(f"Prediction: {pred}")

# Parse the prediction string
try:
    trajectory_points = ast.literal_eval(pred)
except (ValueError, SyntaxError):
    print("Could not parse the prediction string.")
    exit()

# Open the original image
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
width, height = image.size

# Denormalize and draw the trajectory
scaled_points = []
for point in trajectory_points:
    x = int(point[0] * width)
    y = int(point[1] * height)
    scaled_points.append((x, y))
    # Draw a circle at each point
    radius = 5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')

# Draw lines connecting the points
if len(scaled_points) > 1:
    draw.line(scaled_points, fill='blue', width=3)

# Save the result
output_path = "trajectory_output.png"
image.save(output_path)
print(f"Trajectory visualization saved to {output_path}")
