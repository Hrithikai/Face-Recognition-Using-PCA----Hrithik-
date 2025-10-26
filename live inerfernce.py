import cv2
import face_recognition
import numpy as np
import pickle

# Load saved encodings + PCA model
with open("face_encodings_pca.pkl", "rb") as f:
    data = pickle.load(f)

known_face_encodings_pca = data["encodings"]
known_face_names = data["names"]
pca = data["pca"]

TOLERANCE = 0.475

# Start webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Apply PCA on current frame encodings
    if face_encodings:
        face_encodings_pca = pca.transform(face_encodings)

        for (top, right, bottom, left), face_encoding_pca in zip(face_locations, face_encodings_pca):
            distances = np.linalg.norm(known_face_encodings_pca - face_encoding_pca, axis=1)
            matches = distances <= TOLERANCE
            name = "Unknown"

            if any(matches):
                best_match_index = np.argmin(distances)
                name = known_face_names[best_match_index]

            # Draw box + name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
