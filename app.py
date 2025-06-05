import os
import uuid
import sys
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils.classifier import classify_image
from enhancer import enhance_image
from detector import detect_stage

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return jsonify({"message": "🌾 Rice Blast Detection Backend is running!"})

def save_uploaded_image(file):
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".jpg"
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        filename += ".jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return path.replace("\\", "/")

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_path = save_uploaded_image(request.files['image'])
    label, confidence = classify_image(image_path)

    return jsonify({
        "label": label,
        "confidence": f"{confidence:.2f}%",
        "file_saved": image_path
    })

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_path = save_uploaded_image(request.files['image'])
    label, confidence = classify_image(image_path)

    if label != 'RICE LEAF':
        return jsonify({
            'label': label,
            'confidence': f"{confidence:.2f}%",
            'message': '🚫 Not a valid rice leaf. Enhancement aborted.'
        })

    try:
        enhanced_path = enhance_image(image_path, output_dir=OUTPUT_FOLDER)
        return jsonify({
            'label': label,
            'confidence': f"{confidence:.2f}%",
            'enhanced_path': enhanced_path.replace("\\", "/"),
            'status': 'enhanced successfully'
        })
    except Exception as e:
        return jsonify({'error': f'Enhancement failed: {str(e)}'}), 500

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_path = save_uploaded_image(request.files['image'])
    label, confidence = classify_image(image_path)

    if label != 'RICE LEAF':
        return jsonify({
            'label': label,
            'confidence': f"{confidence:.2f}%",
            'message': '🚫 Not a valid rice leaf. Detection aborted.'
        })

    try:
        enhanced_path = enhance_image(image_path, output_dir=OUTPUT_FOLDER)
        output_path, detections = detect_stage(enhanced_path, output_dir=OUTPUT_FOLDER)

        return jsonify({
            'label': label,
            'confidence': f"{confidence:.2f}%",
            'output_image': output_path.replace("\\", "/"),
            'detections': detections,
            'status': 'disease stage detected'
        })

    except Exception as e:
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000

    for arg in sys.argv[1:]:
        if arg.startswith('--host='):
            host = arg.split('=')[1]
        elif arg.startswith('--port='):
            port = int(arg.split('=')[1])

    print(f"\n✅ Backend running! Access it at: http://{host}:{port}/\n")
    app.run(debug=True, host=host, port=port)
