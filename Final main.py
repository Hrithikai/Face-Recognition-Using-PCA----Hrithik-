import cv2
import dlib
import time
from scipy.spatial import distance as dist
from imutils import face_utils
import face_recognition
import numpy as np
import os
from sklearn.decomposition import PCA

# Constants for blink detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
PROXIMITY_THRESHOLD = 50
TOLERANCE = 0.475

# Load pre-trained face and facial landmarks detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\Bio metrics\shape_predictor_68_face_landmarks.dat")

# Load known faces and encodings
known_face_encodings = []
known_face_names = []
dataset_folder = "D:\Bio metrics\Dataset\Faces"

for person_name in os.listdir(dataset_folder):
    person_folder = os.path.join(dataset_folder, person_name)
    if not os.path.isdir(person_folder):
        continue
    for filename in os.listdir(person_folder):
        image_path = os.path.join(person_folder, filename)
        image = face_recognition.load_image_file(image_path)
        try:
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(person_name)
        except IndexError:
            print(f"Face not found in {filename}, skipping.")

# Apply PCA to known face encodings
n_components = min(len(known_face_encodings), len(known_face_encodings[0])) - 1  # Set n_components dynamically
pca = PCA(n_components=n_components)
known_face_encodings_pca = pca.fit_transform(known_face_encodings)

# Initialize variables for liveness detection
persons = {}
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize confusion matrix variables
tp, tn, fp, fn = 0, 0, 0, 0

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_center(rect):
    x, y, w, h = face_utils.rect_to_bb(rect)
    return (x + w // 2, y + h // 2)

def is_same_face(center, persons):
    for face_id, data in persons.items():
        if dist.euclidean(center, data["center"]) < PROXIMITY_THRESHOLD:
            return face_id
    return None

# Start video capture
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        current_persons = {}

        for face in faces:
            face_center = calculate_center(face)
            face_id = is_same_face(face_center, persons)

            if face_id is None:
                face_id = len(current_persons) + 1
                current_persons[face_id] = {
                    "blink_count": 0,
                    "counter": 0,
                    "status": "Unknown",
                    "center": face_center
                }
            else:
                current_persons[face_id] = persons[face_id]

            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                current_persons[face_id]["counter"] += 1
            else:
                if current_persons[face_id]["counter"] >= EYE_AR_CONSEC_FRAMES:
                    current_persons[face_id]["blink_count"] += 1
                    current_persons[face_id]["status"] = "Passed (Live)"
                current_persons[face_id]["counter"] = 0

            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {current_persons[face_id]['blink_count']}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Status: {current_persons[face_id]['status']}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if current_persons[face_id]["status"] == "Passed (Live)":
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                # Apply PCA on the current frame's face encoding
                if face_encodings:
                    face_encodings_pca = pca.transform(face_encodings)

                    for face_encoding_pca in face_encodings_pca:
                        distances = np.linalg.norm(known_face_encodings_pca - face_encoding_pca, axis=1)
                        matches = distances <= TOLERANCE
                        name = "Unknown"
                        if any(matches):
                            best_match_index = np.argmin(distances)
                            name = known_face_names[best_match_index]

                            if name == known_face_names[best_match_index]:
                                tp += 1
                            else:
                                fn += 1
                        else:
                            fp += 1

                        cv2.putText(frame, name, (x, y + h + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        persons = current_persons
        cv2.imshow("Liveness Detection and Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    tn = 0  # Assuming no specific negatives; adjust as needed

    # Metrics calculations
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate

    print(f"\nConfusion Matrix Metrics:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")
    print(f"False Acceptance Rate (FAR): {far:.2f}")
    print(f"False Rejection Rate (FRR): {frr:.2f}")

finally:
    cap.release()
    cv2.destroyAllWindows()