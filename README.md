# Touchless Hand Gesture HCI System using CNN & OpenCV

A real-time Human-Computer Interaction (HCI) application that uses a 6-class **Convolutional Neural Network (CNN)**, **OpenCV**, and **PyAutoGUI** to control system media playback and browser hotkeys using real-time webcam gestures.

---

## 📌 Features
* **Real-Time Gesture Recognition:** Operates at 30+ FPS directly from your webcam.
* **Lighting-Independent Processing:** Employs Otsu's binarization and morphological filtering to isolate clear hand silhouettes.
* **Dual-Key Control Integration:** Triggers both native OS media keys and browser-specific hotkeys (e.g., YouTube shortcuts).
* **Action Debouncing:** Built-in 1.5-second cooldown timer prevents accidental key-spamming when holding a posture.
* **Dedicated Idle State:** Includes a trained `Background` class to keep the system idle when no hand is present in the ROI.

---

## 🖐️ Gesture-to-Action Mapping

| Gesture | Class | Action Executed | Keyboard Shortcut |
| :---: | :--- | :--- | :--- |
| 🛑 | `Background` | Idle (No Action) | None |
| ✊ | `Fist` | Toggle Mute | `Mute` / `'m'` |
| ☝️ | `One` | Increase Master Volume | `Volume Up` |
| ✌️ | `Two` | Decrease Master Volume | `Volume Down` |
| ✋ | `Palm` | Play / Pause Media | `Play-Pause` / `'k'` |
| 👍 | `ThumbsUp` | Next Video / Track | `Next Track` / `Shift + N` |

---

## Project Structure

```text
DL_mini_project/
├── train.py               # Interactive dataset capture & CNN model training script
├── app.py                 # Real-time HCI webcam inference & key execution interface
├── hand_gesture_cnn.h5    # Saved trained CNN model weights
├── gesture_dataset/       # Saved image datasets (categorized by class)
├── .gitignore             # File exclusion rules
└── README.md              # Project documentation
```
---

## Installation & Setup

---
### 1. Clone the Repository
```
git clone https://github.com/Sindhuja-M-309/Hand-gesture-control-cnn.git

cd Hand-gesture-control-cnn
```

### 2. Set Up Virtual Environment
```
python -m venv venv 

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Linux/macOS:
source venv/bin/activate
```

---

## How to Run
Run Real-Time Gesture Control

Launch the main HCI control application:
```
python app.py
```

1.Position your hand inside the blue ROI box in the top-right corner of the camera feed.

2.Open YouTube or Spotify in your browser, click once on the media player to focus, and control playback with hand gestures.

3.Press q while focused on the camera window to quit.

---

## Retrain Model (Optional)

To capture fresh training data and retrain the network:

```
python train.py
```

Press s to begin capturing 300 samples for each class (Background, Fist, One, Two, Palm, ThumbsUp), or q to abort.

---

## Model Architecture & Performance

**Input Resolution:** 64 x 64 x 1 (Grayscale Binary Mask)

**Architecture:** 3x Conv2D + BatchNormalization + MaxPooling2D + Dropout (0.25) → Dense (128, Dropout 0.5) → Softmax Output (6 classes)

**Training Accuracy:** ~96.4%

**Validation Accuracy:** ~97.7%

**Inference Latency:** ~18ms per frame

---

### License

This project is licensed under the MIT License - see the LICENSE file for details.

---
