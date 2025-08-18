from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import cv2
import time

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
ANNOTATION_FOLDER = 'annotations'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATION_FOLDER'] = ANNOTATION_FOLDER

# Create upload and annotation folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATION_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'})
    files = request.files.getlist('files[]')
    for file in files:
        if file.filename == '':
            continue
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({'message': 'Files uploaded successfully'})

@app.route('/images')
def get_images():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify(images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    filename = data['filename']
    trajectory = data['trajectory']
    instruction = data['instruction']
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    annotation_data = {
        "id": int(time.time()),
        "meta_data": {
            "original_dataset": "bridge",
            "original_width": width,
            "original_height": height
        },
        "instruction": instruction,
        "points": trajectory
    }

    annotations_file = os.path.join(app.config['ANNOTATION_FOLDER'], 'annotations.json')
    
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            all_annotations = json.load(f)
    else:
        all_annotations = {}

    all_annotations[filename] = annotation_data

    with open(annotations_file, 'w') as f:
        json.dump(all_annotations, f, indent=4)
        
    return jsonify({'message': 'Annotation saved successfully'})

@app.route('/get_annotation/<filename>')
def get_annotation(filename):
    annotations_file = os.path.join(app.config['ANNOTATION_FOLDER'], 'annotations.json')
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            all_annotations = json.load(f)
        return jsonify(all_annotations.get(filename, {}))
    return jsonify({})

if __name__ == '__main__':
    app.run(debug=True)
