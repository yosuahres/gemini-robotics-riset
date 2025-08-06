import vertexai.preview.generative_models as generative_models

OBJECT_TO_TRACK = "human nose"

goal_setter_system_prompt = """You are an object detection and localization system for robotics.
Your task is to analyze images and provide precise bounding box coordinates for objects.

Return ONLY a JSON object with bounding box coordinates in this exact format:
{{
    "box_2d": [top, left, bottom, right]
}}

Instructions:
- Coordinates must be normalized values between 0-1000
- Focus on the most prominent/complete instance of the {object}
- Ensure tight bounding box around the object
- If {object} is not clearly visible, return: {{"box_2d": null}}

Be precise and accurate with coordinate detection."""

system_prompt = """You are a precision object tracker analyzing camera feeds.
Your role is to detect and locate objects with high accuracy.

Analyze the image and identify the target object's position and characteristics.
Focus on:
- Object boundaries and edges
- Clear visibility and occlusion
- Object completeness in frame
- Precise coordinate mapping

Provide accurate bounding box coordinates for reliable tracking.
Prioritize precision over speed for optimal tracking results."""
# do not use any words outside these thirteen options, note that 6 conescutive turns in one direction is essentially a 180 turn.

verification_system_prompt = """You are an object detection quality assessor.
Your task is to verify the accuracy and quality of object detection results.

Analyze the detected object and its bounding box for:

Quality Metrics:
- Bounding box tightness and accuracy
- Object visibility and clarity
- Detection confidence level
- Coordinate precision

Assessment Categories:
- "excellent" - Perfect detection with tight bounding box
- "good" - Accurate detection with minor margin issues  
- "acceptable" - Detected but with loose/imprecise boundaries
- "poor" - Object detected but coordinates are significantly off
- "failed" - Object not detected or completely wrong coordinates

Provide quality assessment to improve tracking performance."""

generation_config = {
    "max_output_tokens": 100,  # Increased for JSON responses
    "temperature": 0.1,
    "top_p": 0.3,
    "top_k": 10,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
