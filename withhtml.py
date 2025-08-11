import os
import json
import argparse
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from google import genai
from google.genai import types

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.5-pro"
DEFAULT_IMAGE_PATH = "image-testing/kettlelab.png"

def initialize_client(api_key):
    return genai.Client(api_key=api_key)

def load_and_resize_image(image_path, size=(800, 800)):
    try:
        img = Image.open(image_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return img
    except FileNotFoundError:
        return None

def get_points_from_image(client, model_id, image, prompt):
    response = client.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    if not response.text:
        print("Error: Received an empty response from the API.")
        if response.prompt_feedback:
            print(f"Prompt feedback: {response.prompt_feedback}")
        return []
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        fixed_json = "[" + response.text.replace("}\n{", "},{") + "]"
        return json.loads(fixed_json)

def draw_points(image, points):
    draw = ImageDraw.Draw(image)
    for item in points:
        point = item['point']
        x = int(point[1] / 1000.0 * image.width)
        y = int(point[0] / 1000.0 * image.height)

        radius = 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')
        draw.text((x + 10, y), item['label'], fill='red')

def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:]) 
            json_output = json_output.split("```")[0] 
            break 
    return json_output

def generate_point_html(pil_image, points_data):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    points_json = json.dumps(points_data)

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Point Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #fff;
            color: #000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
            opacity: 0;
            transition: all 0.3s ease-in;
            pointer-events: auto;
        }}

        .point.visible {{
            opacity: 1;
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
            font-size: 14px;
            padding: 4px 12px;
            border-radius: 4px;
            transform: translate(20px, -10px);
            white-space: nowrap;
            opacity: 0;
            transition: all 0.3s ease-in;
            box-shadow: 0 0 30px rgba(41, 98, 255, 0.4);
            pointer-events: auto;
            cursor: pointer;
        }}

        .point-label.visible {{
            opacity: 1;
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
    </style>
</head>
<body>
    <div id="container" style="position: relative;">
        <canvas id="canvas" style="background: #000;"></canvas>
        <div id="pointOverlay" class="point-overlay"></div>
    </div>

    <script>
        function annotatePoints(frame) {{
            // Add points with fade effect
            const pointsData = {points_json};

            const pointOverlay = document.getElementById('pointOverlay');
            pointOverlay.innerHTML = '';

            const points = [];
            const labels = [];

            pointsData.forEach(pointData => {{
                // Skip entries without coodinates.
                if (!(pointData.hasOwnProperty("point")))
                  return;

                const point = document.createElement('div');
                point.className = 'point';
                const [y, x] = pointData.point;
                point.style.left = `${{x/1000.0 * 100.0}}%`;
                point.style.top = `${{y/1000.0 * 100.0}}%`;

                const pointLabel = document.createElement('div');
                pointLabel.className = 'point-label';
                pointLabel.textContent = pointData.label;
                point.appendChild(pointLabel);

                pointOverlay.appendChild(point);
                points.push(point);
                labels.push(pointLabel);

                setTimeout(() => {{
                    point.classList.add('visible');
                    pointLabel.classList.add('visible');
                }}, 0);

                // Add hover effects
                const handleMouseEnter = () => {{
                    // Highlight current point and label
                    point.classList.add('highlight');
                    pointLabel.classList.add('highlight');

                    // Fade out other points and labels
                    points.forEach((p, idx) => {{
                        if (p !== point) {{
                            p.classList.add('fade-out');
                            labels[idx].classList.add('fade-out');
                        }}
                    }});
                }};

                const handleMouseLeave = () => {{
                    // Remove highlight from current point and label
                    point.classList.remove('highlight');
                    pointLabel.classList.remove('highlight');

                    // Restore other points and labels
                    points.forEach((p, idx) => {{
                        p.classList.remove('fade-out');
                        labels[idx].classList.remove('fade-out');
                    }});
                }};

                point.addEventListener('mouseenter', handleMouseEnter);
                point.addEventListener('mouseleave', handleMouseLeave);
                pointLabel.addEventListener('mouseenter', handleMouseEnter);
                pointLabel.addEventListener('mouseleave', handleMouseLeave);
            }});
        }}

        // Initialize canvas
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const container = document.getElementById('container');

        // Load and draw the image
        const img = new Image();
        img.onload = () => {{
            const aspectRatio = img.height / img.width;
            canvas.width = 800;
            canvas.height = Math.round(800 * aspectRatio);
            container.style.width = canvas.width + 'px';
            container.style.height = canvas.height + 'px';

            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            frame.width = canvas.width;
            frame.height = canvas.height;
            annotatePoints(frame);
        }};
        img.src = 'data:image/png;base64,{img_str}';

        const frame = {{
            width: canvas.width,
            height: canvas.height
        }};

        annotatePoints(frame);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="analisis input")
    parser.add_argument("image_path", type=str, nargs='?', default=DEFAULT_IMAGE_PATH,
                        help=f"The path to the image file (default: {DEFAULT_IMAGE_PATH})")
    args = parser.parse_args()

    try:
        client = initialize_client(API_KEY)
    except ValueError as e:
        print(e)
        exit()

    print(f"Using model: {MODEL_ID}")
    pointing_image = load_and_resize_image(args.image_path)

    if pointing_image:
        pointing_prompt = """
          Pinpoint switches or controls on the device.
          Switches or controls can be buttons, knobs, levers, dials, touchscreens, or any other interactive element.
          The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000. One element a line.
          Explain how to use each part, put them in the label field, remove duplicated parts and instructions.
        """
        try:
            points = get_points_from_image(client, MODEL_ID, pointing_image.copy(), pointing_prompt)
            print("Pointing Results:")
            for item in points:
                print(f"  - Label: {item['label']}, Point: {item['point']}")
            
            pointing_image_with_results = pointing_image.copy()
            draw_points(pointing_image_with_results, points)
            pointing_image_with_results.convert('RGB').save("results/pointing_results.jpg")
            print("Pointing results saved to results/pointing_results.jpg")
            
            html_content = generate_point_html(pointing_image.copy(), points)
            with open("results/pointing_results.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            print("Interactive HTML visualization saved to results/pointing_results.html")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing pointing response: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
