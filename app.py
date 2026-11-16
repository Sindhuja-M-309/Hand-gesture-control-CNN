import time
import cv2
import numpy as np
import pyautogui
import tensorflow as tf

# Set PyAutoGUI delay
pyautogui.PAUSE = 0.1

# Configuration
IMG_SIZE = 64
CLASSES = ['Background', 'Fist', 'One', 'Two', 'Palm', 'ThumbsUp']

# Load Trained Model
model = tf.keras.models.load_model('hand_gesture_cnn.h5')

# Cooldown mechanism to prevent rapid key repetition
last_action_time = 0
COOLDOWN_DELAY = 1.5  # Seconds between triggering actions


def preprocess_frame(roi_frame):
    """Processes ROI frame with Otsu thresholding & morphological filtering."""
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(
        blur, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Clean isolated noise pixels and separate digit gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresholded = cv2.morphologyEx(
        thresholded, cv2.MORPH_OPEN, kernel, iterations=1
    )

    resized = cv2.resize(thresholded, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    tensor = np.expand_dims(normalized, axis=(0, -1))
    return tensor, thresholded


def execute_action(gesture):
    """Maps recognized gestures with OS-level media keys (works without focus)."""
    global last_action_time
    current_time = time.time()

    if gesture == 'Background':
        return

    if current_time - last_action_time < COOLDOWN_DELAY:
        return

    if gesture == 'Fist':
        # Mute: Sends OS Mute + YouTube 'm'
        pyautogui.press('volumemute')
        pyautogui.press('m')
        print("Action Executed: Toggle Mute")
        last_action_time = current_time

    elif gesture == 'One':
        # Master Volume Up
        pyautogui.press('volumeup')
        print("Action Executed: Volume Up")
        last_action_time = current_time

    elif gesture == 'Two':
        # Master Volume Down
        pyautogui.press('volumedown')
        print("Action Executed: Volume Down")
        last_action_time = current_time

    elif gesture == 'Palm':
        # Play/Pause: Sends OS Play/Pause key + YouTube 'k' key
        pyautogui.press('playpause')
        pyautogui.press('k')
        print("Action Executed: Play / Pause")
        last_action_time = current_time

    elif gesture == 'ThumbsUp':
        # Next Video: Sends OS Next Track key + YouTube Shift+N key
        pyautogui.press('nexttrack')
        pyautogui.hotkey('shift', 'n')
        print("Action Executed: Next Video / Track")
        last_action_time = current_time


# Initialize Webcam Feed
cap = cv2.VideoCapture(0)

print("Starting HCI Gesture Control Interface...")
print("Press 'q' in the camera window to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror frame
    roi = frame[100:350, 300:550]

    # Preprocess & Predict
    input_tensor, processed_roi = preprocess_frame(roi)
    predictions = model.predict(input_tensor, verbose=0)

    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    gesture = CLASSES[class_idx]

    # Display HUD Feedback
    if gesture != 'Background':
        display_text = f"Gesture: {gesture} ({confidence*100:.1f}%)"
        cv2.putText(
            frame,
            display_text,
            (300, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        if confidence > 0.85:
            execute_action(gesture)
    else:
        cv2.putText(
            frame,
            "Status: Idle (No Gesture)",
            (300, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # UI ROI Box
    cv2.rectangle(frame, (300, 100), (550, 350), (255, 0, 0), 2)
    cv2.imshow("HCI Main Interface", frame)
    cv2.imshow("Binary Hand ROI", processed_roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()