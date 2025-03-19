from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__)

# Load Face Detection & Recognition Model
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

@app.route('/')
def home():
    return "Welcome to the Face Recognition API! Use the /recognize endpoint to upload an image."

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        # Check if an image file was sent
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        image = Image.open(file)  # Open the image using PIL
        
        # Convert image to RGB and NumPy array
        image = image.convert('RGB')
        img_array = np.array(image)

        # Detect face using MTCNN
        boxes, _ = mtcnn.detect(img_array)
        if boxes is None:
            return jsonify({"error": "No face detected"}), 400
        
        # Extract the first face
        x1, y1, x2, y2 = map(int, boxes[0])
        face_crop = img_array[y1:y2, x1:x2]

        # Convert face to PIL Image and resize
        face = Image.fromarray(face_crop).resize((160, 160))
        face_tensor = torch.tensor(np.array(face)).permute(2, 0, 1).float().div(255).unsqueeze(0)

        # Generate embeddings using FaceNet
        embedding = resnet(face_tensor).detach().numpy().tolist()

        return jsonify({"face_embedding": embedding})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
