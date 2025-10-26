import cv2
import dlib
import time
from scipy.spatial import distance as dist
from imutils import face_utils

# EAR calculation function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for blink detection
EYE_AR_THRESH = 0.25  # Threshold for eye aspect ratio to indicate blink
EYE_AR_CONSEC_FRAMES = 3  # Consecutive frames to indicate a blink
PROXIMITY_THRESHOLD = 50  # Threshold for proximity to avoid duplicate face detections

# Load pre-trained face and facial landmarks detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\Bio metrics\shape_predictor_68_face_landmarks.dat")

# Get the indexes for the left and right eye from the facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize variables for each person detected
persons = {}

# Start the webcam video stream
cap = cv2.VideoCapture(0)

def calculate_center(rect):
    """Calculate the center of the face bounding box."""
    x, y, w, h = face_utils.rect_to_bb(rect)
    return (x + w // 2, y + h // 2)

def is_same_face(center, persons):
    """Check if a face is the same as an existing one based on proximity."""
    for face_id, data in persons.items():
        if dist.euclidean(center, data["center"]) < PROXIMITY_THRESHOLD:
            return face_id
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop through each face detected
    for face in faces:
        # Calculate the center of the face bounding box
        face_center = calculate_center(face)

        # Check if the face is already being tracked
        face_id = is_same_face(face_center, persons)
        
        # Assign a new ID if this face is not close to any tracked face
        if face_id is None:
            face_id = len(persons) + 1
            persons[face_id] = {
                "blink_count": 0,
                "counter": 0,
                "status": "Unknown",
                "center": face_center
            }
        else:
            # Update the center for tracking movement
            persons[face_id]["center"] = face_center

        # Get the facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the eye aspect ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratio (EAR)
        ear = (leftEAR + rightEAR) / 2.0

        # Check if the EAR is below the blink threshold, indicating a blink
        if ear < EYE_AR_THRESH:
            persons[face_id]["counter"] += 1
        else:
            # If the eyes were below the threshold for a sufficient number of frames, register a blink
            if persons[face_id]["counter"] >= EYE_AR_CONSEC_FRAMES:
                persons[face_id]["blink_count"] += 1
                persons[face_id]["status"] = "Passed (Live)"  # Update status immediately after a blink
            persons[face_id]["counter"] = 0

        # Draw the face bounding box and blink count on the frame
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {persons[face_id]['blink_count']}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {persons[face_id]['status']}", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with face boxes and liveness status
    cv2.imshow("Liveness Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
