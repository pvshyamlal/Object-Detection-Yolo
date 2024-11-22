from flask import Flask, render_template, request, send_from_directory
import os
from run_yolo import run_yolo
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded'
app.config['DETECTED_FOLDER'] = 'static/detected'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Run YOLO detection
        run_yolo(app.config['UPLOAD_FOLDER'], app.config['DETECTED_FOLDER'])

        detected_file_path = os.path.join(app.config['DETECTED_FOLDER'], file.filename)
        if not Path(detected_file_path).exists():
            return "Detection failed", 500

        # Return the relative URL to the detected image
        relative_path = f'detected/{file.filename}'
        return relative_path, 200

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)
if __name__ == "__main__":
    app.run(debug=True)
