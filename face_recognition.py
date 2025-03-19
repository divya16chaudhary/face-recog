import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

face_detector = MTCNN(keep_all=True)
face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()

def extract_face_embeddings(img):
    with torch.no_grad():
        face_boxes, _ = face_detector.detect(img)
        if face_boxes is not None:
            embeddings = []
            for bbox in face_boxes:
                x1, y1, x2, y2 = map(int, bbox)
                face_crop = img[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue
                
                face_resized = cv2.resize(face_crop, (160, 160))
                face_norm = np.transpose(face_resized, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face_norm).unsqueeze(0)
                
                
                embedding = face_recognizer(face_tensor).detach().numpy().flatten()
                embeddings.append(embedding)

            return embeddings
    return []


def register_faces(face_data):
    stored_encodings, stored_names = [], []

    for person_name, image_path in face_data.items():
        img = cv2.imread(image_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encoding = extract_face_embeddings(img_rgb)

            if face_encoding:
                stored_encodings.append(face_encoding[0])  
                stored_names.append(person_name)

    return stored_encodings, stored_names


face_db = {
    "Divya Chaudhary": "images/divya.jpg",
    
}

face_encodings, face_names = register_faces(face_db)
def match_faces(known_encodings, known_names, new_encodings, match_threshold=0.6):
    identified_names = []

    for new_encoding in new_encodings:
        differences = np.linalg.norm(known_encodings - new_encoding, axis=1)
        best_match_idx = np.argmin(differences)

        if differences[best_match_idx] < match_threshold:
            identified_names.append(known_names[best_match_idx])
        else:
            identified_names.append('Unknown')

    return identified_names

# Initialize webcam capture
camera = cv2.VideoCapture(0)
recognition_threshold = 0.6

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    test_encodings = extract_face_embeddings(frame_rgb)

    if test_encodings and face_encodings:
        detected_names = match_faces(np.array(face_encodings), face_names, test_encodings, recognition_threshold)
        
     
        detected_faces, _ = face_detector.detect(frame_rgb)
        if detected_faces is not None:
            for label, bbox in zip(detected_names, detected_faces):
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Live Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
