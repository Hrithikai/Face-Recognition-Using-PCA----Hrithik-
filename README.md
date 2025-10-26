# Face-Recognition-Using-PCA with Liveness Detection----Hrithik-
This project implements facial recognition with anti-spoofing capabilities through blink detection. It uses PCA for dimensionality reduction to improve recognition speed and accuracy.


## 🚀 Overview
This project is a **real-time facial recognition system** with **blink-based liveness detection**, built using **Python 3.9.6**.  
It integrates **PCA for dimensionality reduction**, **EAR-based blink detection**, and **multi-face tracking** for anti-spoofing verification.  
Designed for **security systems**, **access control**, and **AI-driven biometric authentication**.

---

## 🧠 Core Architecture
- **Facial Recognition:** Uses PCA-compressed facial encodings for optimized matching.
- **Liveness Detection:** Blink detection using Eye Aspect Ratio (EAR) to prevent spoofing.
- **Real-Time Inference:** Processes webcam input with dlib’s 68-face landmarks model.
- **Performance Metrics:** Tracks accuracy, recall, precision, FAR, and FRR.

---

## 📁 Project Structure

```bash
Facial_Recognition_Liveness/
│
├── data to vectors.py          # Converts face images to PCA-compressed encodings
├── live inerfernce.py          # Performs live face recognition using PCA encodings
├── Blink detection.py          # Detects blinks using Eye Aspect Ratio (EAR)
├── Final main.py               # Integrated face + liveness detection
├── requirements.txt            # Python dependencies
└── Dataset/
    └── Faces/
        ├── Person1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── Person2/
            ├── image1.jpg
            └── image2.jpg
```

---

## ⚙️ Setup & Installation

### 🧩 Prerequisites
- Python 3.9.6  
- Webcam  
- VS Studio Installer (You may need to install some dependencies from it)
- CMAKE (This is will cause some issues, try to resolve it using LLM's help)
- dlib model: `shape_predictor_68_face_landmarks.dat`

| Component | Requirement |
|------------|--------------|
| **Python** | 3.9.6 (Mandatory) |
| **RAM** | Minimum 8GB |
| **CPU** | Minimum 4 cores 8 threads CPU (Recommended 6 cores 12 threads CPU) |
| **GPU** | Any thing is fine ,i have used integrated GPU (Code has been optimized for CPU) |
| **Storage** | Minimum 5GB free space |

### 🛠️ Installation
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Step 1: Generate Face Encodings
```bash
python "data to vectors.py"
```
This creates `face_encodings_pca.pkl` containing PCA-compressed face encodings.

### Step 2: Run Recognition
**Option A – Pre-computed Encodings (Fast Mode)**
```bash
python "live inerfernce.py"
```

**Option B – Integrated with Liveness Detection**
```bash
python "Final main.py"
```

**Option C – Blink Detection Only**
```bash
python "Blink detection.py"
```

---

## 🔑 Key Features
- ⚡ **PCA Compression:** Faster encoding comparisons  
- 👁️ **Blink Detection:** EAR-based liveness verification  
- 👤 **Multi-Face Tracking:** Tracks and verifies multiple faces  
- 🧮 **Real-Time Performance:** Optimized for smooth webcam inference  

---

## 🔧 Configuration Parameters

| Parameter | Description | Default Value |
|------------|--------------|----------------|
| `EYE_AR_THRESH` | Blink detection sensitivity | 0.25 |
| `TOLERANCE` | Face recognition tolerance | 0.475 |
| `PROXIMITY_THRESHOLD` | Face tracking distance | 50 |

---

## 🧾 Controls
| Key | Action |
|------|---------|
| `q` | Quit the application |
| `Ctrl + C` | Show performance metrics in Final main.py |

---

## 📊 Performance Metrics
At program exit, `Final main.py` displays key metrics:
- True Positives (TP), False Positives (FP)  
- Accuracy, Precision, Recall, F1-Score  
- False Acceptance Rate (FAR), False Rejection Rate (FRR)

---

## 💡 Notes
- Update file paths to match your system setup.  
- Ensure proper lighting for best facial feature detection.  
- The system marks a user as **“Passed (Live)”** only after detecting a blink.  

---

## 📸 Add a Project Image
To include your own image or demo screenshot:
1. Create a folder named `assets` in your repo root.  
2. Add your image as `project_banner.png` (recommended size: 1200×400).  
3. Replace the banner image path in this README.

---

## 🧾 License
This project is released under the [MIT License](LICENSE).

---

**Developed with ❤️ using Python & OpenCV**
