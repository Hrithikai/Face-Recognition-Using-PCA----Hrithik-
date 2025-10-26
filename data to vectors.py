import os
import face_recognition
import numpy as np
import pickle
from sklearn.decomposition import PCA

# Path to dataset
dataset_folder = r"D:\Bio metrics\Dataset\Faces"
output_file = "face_encodings_pca.pkl"

known_face_encodings = []
known_face_names = []

# Load dataset images and generate encodings
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
            print(f"[!] Face not found in {filename}, skipping.")

# Apply PCA for dimensionality reduction
if known_face_encodings:
    n_components = min(len(known_face_encodings), len(known_face_encodings[0])) - 1
    pca = PCA(n_components=n_components)
    known_face_encodings_pca = pca.fit_transform(known_face_encodings)

    # Save encodings + PCA model + names
    with open(output_file, "wb") as f:
        pickle.dump({
            "encodings": known_face_encodings_pca,
            "names": known_face_names,
            "pca": pca
        }, f)

    print(f"[✓] Encodings saved to {output_file}")
else:
    print("[x] No faces found in dataset!")
