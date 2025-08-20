import json
import json
import os
from google import genai
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from gemini_config import trajectory_prompt, generation_config, safety_settings

def plan_trajectory(task, image_path=None):
    """
    Generates a trajectory plan based on a high-level task description.

    Args:
        task (str): The high-level task description.
        image_path (str, optional): Path to an image file for visual context.

    Returns:
        list: A list of tuples representing the trajectory plan.
    """
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    
    prompt = trajectory_prompt.format(task=task)
    
    # Prepare content for the API call
    contents = [prompt]
    
    # Add image if provided
    if image_path and os.path.exists(image_path):
        try:
            # Load and resize image for efficiency
            image = Image.open(image_path)
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            contents.append(image)
            print(f"✓ Using image: {image_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load image {image_path}: {e}")
            print("   Proceeding without image context...")
    elif image_path:
        print(f"⚠️  Warning: Image file not found: {image_path}")
        print("   Proceeding without image context...")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=8192,
                temperature=0.2
            )
        )
    except Exception as api_error:
        if "429" in str(api_error) or "RESOURCE_EXHAUSTED" in str(api_error):
            print("⚠️  API quota exceeded. Cannot generate trajectory.")
            return None
        else:
            print(f"⚠️  API error: {api_error}")
            return None

    # Check if response is valid
    if not response:
        print("Error: No response received from API")
        return None
    
    # Extract text from response, handling different response structures
    response_text = None
    
    if hasattr(response, 'text') and response.text is not None:
        response_text = response.text
    elif hasattr(response, 'candidates') and response.candidates:
        # Try to get text from the first candidate
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                # Extract text from parts
                text_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    response_text = ''.join(text_parts)
            elif hasattr(candidate.content, 'text') and candidate.content.text:
                response_text = candidate.content.text
    
    if response_text is None:
        print("Error: Could not extract text from response")
        if hasattr(response, 'prompt_feedback'):
            print(f"Prompt feedback: {response.prompt_feedback}")
        if hasattr(response, 'candidates'):
            print(f"Candidates: {response.candidates}")
            if response.candidates:
                candidate = response.candidates[0]
                print(f"First candidate content: {candidate.content}")
                if hasattr(candidate.content, 'parts'):
                    print(f"Content parts: {candidate.content.parts}")
        return None

    print(f"✓ Received response from API")
    print(f"Full response: {response_text}")

    try:
        # Clean the response text in case it has markdown formatting
        clean_response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if clean_response_text.startswith("```json"):
            clean_response_text = clean_response_text[7:]
        if clean_response_text.endswith("```"):
            clean_response_text = clean_response_text[:-3]
        
        clean_response_text = clean_response_text.strip()
        
        # Try to extract just the list part if it's embedded in text
        import re
        list_pattern = r'\[\[.*?\]\]'
        match = re.search(list_pattern, clean_response_text, re.DOTALL)
        if match:
            list_text = match.group()
            print(f"Extracted list: {list_text}")
            trajectory_points = json.loads(list_text)
            return trajectory_points
        
        # If no list found, try parsing the whole response as JSON
        trajectory_points = json.loads(clean_response_text)
        return trajectory_points
        
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error parsing model response: {e}")
        print(f"Attempting to extract coordinates manually...")
        
        # Manual extraction as fallback
        try:
            import re
            # Look for coordinate patterns like [x, y] or (x, y)
            coord_pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
            coordinates = re.findall(coord_pattern, response_text)
            if coordinates:
                trajectory_points = [[float(x), float(y)] for x, y in coordinates]
                print(f"Manually extracted {len(trajectory_points)} coordinate pairs")
                return trajectory_points
        except Exception as manual_error:
            print(f"Manual extraction also failed: {manual_error}")
        
        print("⚠️  All parsing methods failed. No trajectory generated.")
        return None


def parse_trajectory_response(response_text):
    """Parse trajectory points from API response text."""
    try:
        # Clean the response text
        clean_text = response_text.strip()
        
        # Remove markdown code blocks
        if clean_text.startswith("```"):
            lines = clean_text.split('\n')
            clean_text = '\n'.join(lines[1:-1])
        
        clean_text = clean_text.strip()
        
        # Try to extract coordinate list patterns
        import re
        
        # Pattern 1: [[x1,y1],[x2,y2],...]
        list_pattern = r'\[\s*\[\s*[\d.]+\s*,\s*[\d.]+\s*\](?:\s*,\s*\[\s*[\d.]+\s*,\s*[\d.]+\s*\])*\s*\]'
        match = re.search(list_pattern, clean_text)
        
        if match:
            list_text = match.group()
            print(f"Found coordinate list: {list_text}")
            try:
                trajectory_points = json.loads(list_text)
                if isinstance(trajectory_points, list) and len(trajectory_points) > 0:
                    return trajectory_points
            except json.JSONDecodeError:
                pass
        
        # Pattern 2: Individual coordinate pairs
        coord_pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        coordinates = re.findall(coord_pattern, clean_text)
        
        if coordinates:
            trajectory_points = [[float(x), float(y)] for x, y in coordinates]
            print(f"Extracted {len(trajectory_points)} coordinate pairs")
            return trajectory_points
        
        # Pattern 3: Try parsing as direct JSON
        try:
            trajectory_points = json.loads(clean_text)
            if isinstance(trajectory_points, list):
                return trajectory_points
        except json.JSONDecodeError:
            pass
        
        return None
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None


def draw_trajectory_on_image(image_path, trajectory_points, output_path="results/trajectory_output.jpg"):
    """
    Draw trajectory points and path on the image and save it.
    
    Args:
        image_path (str): Path to the original image
        trajectory_points (list): List of [x, y] coordinate pairs
        output_path (str): Path to save the output image
    """
    try:
        # Load the original image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not trajectory_points or len(trajectory_points) == 0:
            print("No trajectory points to draw")
            return None
        
        # Convert trajectory points to image coordinates
        img_width, img_height = image.size
        image_coords = []
        
        for i, point in enumerate(trajectory_points):
            if len(point) >= 2:
                # Assume points are normalized 0-1, convert to image coordinates
                x = point[0] * img_width if point[0] <= 1.0 else point[0]
                y = point[1] * img_height if point[1] <= 1.0 else point[1]
                
                # Clamp to image bounds
                x = max(0, min(img_width - 1, x))
                y = max(0, min(img_height - 1, y))
                
                image_coords.append((x, y))
        
        if len(image_coords) < 2:
            print("Need at least 2 points to draw trajectory")
            return None
        
        # Draw trajectory path (lines between points)
        for i in range(len(image_coords) - 1):
            start_point = image_coords[i]
            end_point = image_coords[i + 1]
            
            # Draw line with gradient color (blue to red)
            color_ratio = i / max(1, len(image_coords) - 2)
            red = int(255 * color_ratio)
            blue = int(255 * (1 - color_ratio))
            line_color = (red, 0, blue)
            
            draw.line([start_point, end_point], fill=line_color, width=3)
        
        # Draw trajectory points
        for i, coord in enumerate(image_coords):
            x, y = coord
            
            # Different colors and sizes for start, middle, and end points
            if i == 0:
                # Start point - green circle
                radius = 4
                color = (0, 255, 0)  # Green
                label = "START"
            elif i == len(image_coords) - 1:
                # End point - red circle
                radius = 4
                color = (255, 0, 0)  # Red
                label = "END"
            else:
                # Middle points - yellow circles
                radius = 2
                color = (255, 255, 0)  # Yellow
                label = str(i)
            
            # Draw circle
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                        fill=color, outline=(0, 0, 0), width=2)
        
        # Add title
        title = f"Trajectory Plan ({len(trajectory_points)} points)"
        try:
            title_font = ImageFont.load_default()
            draw.text((10, 10), title, fill=(255, 255, 255), font=title_font)
            draw.text((10, 30), f"Path: {os.path.basename(image_path)}", fill=(255, 255, 255), font=title_font)
        except:
            draw.text((10, 10), title, fill=(255, 255, 255))
        
        # Save the output image
        image.save(output_path, quality=95)
        print(f"✓ Trajectory visualization saved to: {output_path}")
        
        # Also save trajectory data as JSON
        json_output_path = output_path.replace('.jpg', '.json').replace('.png', '.json')
        with open(json_output_path, 'w') as f:
            json.dump({
                "source_image": image_path,
                "trajectory_points": trajectory_points,
                "image_coordinates": [[x, y] for x, y in image_coords],
                "point_count": len(trajectory_points)
            }, f, indent=2)
        print(f"✓ Trajectory data saved to: {json_output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error drawing trajectory: {e}")
        return None


if __name__ == '__main__':
    print("🤖 Trajectory Planning with Image")
    print("=" * 40)
    
    # Use a single image for trajectory planning
    image_path = "image-testing/door2.jpg"
    task_description = "operate lever door handle by pressing down to open door"
    
    print(f"Task: {task_description}")
    print(f"Image: {image_path}")
    print()
    
    trajectory_plan = plan_trajectory(task_description, image_path)
    
    if trajectory_plan:
        print("✓ Trajectory plan generated successfully!")
        print(json.dumps(trajectory_plan, indent=2))
        
        # Draw the trajectory on the image
        print("\n📊 Generating trajectory visualization...")
        output_image = draw_trajectory_on_image(
            image_path, 
            trajectory_plan, 
            "results/door_trajectory_visualization.jpg"
        )
        
        if output_image:
            print(f"🎨 Trajectory visualization complete!")
            print(f"   View the result at: {output_image}")
        else:
            print("⚠️  Could not create trajectory visualization")
            
    else:
        print("✗ Failed to generate trajectory plan")
