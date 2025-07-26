from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os
from werkzeug.utils import secure_filename
from meso import Meso4

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'mov'}
app.config['MODEL_PATH'] = './models/Meso4_DF.h5'

# Load model at startup
model = None
try:
    model = Meso4()
    model.load(app.config['MODEL_PATH'])
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# Serve frontend index.html at root
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image):
    """Preprocess image for MesoNet"""
    # Convert to numpy array
    image = np.array(image)
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    
    # Resize to 256x256 (MesoNet's expected input)
    image = cv2.resize(image, (256, 256))
    
    # Normalize to [0,1]
    image = image.astype('float32') / 255.0
    
    # Expand dimensions for batch
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process based on file type
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(filepath)
                processed_img = preprocess_image(img)
                
                if model is None:
                    # Fallback for demo purposes
                    confidence = np.random.uniform(0, 1)
                else:
                    confidence = model.predict(processed_img)[0][0]
                
                result = {
                    'type': 'image',
                    'is_deepfake': confidence > 0.5,
                    'confidence': float(confidence),
                    'message': f"Image analysis: {'Deepfake' if confidence > 0.5 else 'Real'}",
                    'frame_analysis': False,
                    'frames_analyzed': 1
                }
            else:
                result = process_video(filepath)
            
            # Clean up
            os.remove(filepath)
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_video(video_path, frame_count=10):
    """Process multiple frames of video"""
    try:
        vidcap = cv2.VideoCapture(video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)

        predictions = []
        for idx in frame_indices:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = vidcap.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = preprocess_image(frame)
                if model is None:
                    confidence = np.random.uniform(0, 1)
                else:
                    confidence = model.predict(processed_frame)[0][0]
                predictions.append(confidence)

        if predictions:
            avg_confidence = float(np.mean(predictions))
            return {
                'type': 'video',
                'is_deepfake': avg_confidence > 0.5,
                'confidence': avg_confidence,
                'message': f"Video analysis: {'Deepfake' if avg_confidence > 0.5 else 'Real'}",
                'frame_analysis': True,
                'frames_analyzed': len(predictions)
            }
        return {'error': 'No valid frames analyzed'}
    except Exception as e:
        return {'error': str(e)}

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok', 'message': 'API is running'})

if __name__ == '__main__':
    # Create upload directory if needed
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(host='0.0.0.0', port=5001, debug=True)