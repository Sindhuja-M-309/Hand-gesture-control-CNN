import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.model_selection import train_test_split

# Global Configuration (Includes 6th 'Background' class)
IMG_SIZE = 64
CLASSES = ['Background', 'Fist', 'One', 'Two', 'Palm', 'ThumbsUp']
DATASET_PATH = 'gesture_dataset'
MODEL_SAVE_PATH = 'hand_gesture_cnn.h5'
SAMPLES_PER_CLASS = 300


def create_dataset():
    """Captures gesture samples interactively using the webcam feed."""
    cap = cv2.VideoCapture(0)

    # Pre-create directory structures for all categories
    for category in CLASSES:
        os.makedirs(os.path.join(DATASET_PATH, category), exist_ok=True)

    print("==================================================")
    print("        HAND GESTURE DATASET CAPTURE MODE         ")
    print("==================================================")
    print("NOTE: For 'Background', leave the ROI box EMPTY!")

    for category in CLASSES:
        print(f"\n---> Get ready to record class: '{category}'")
        print("Press 's' to start recording, or 'q' to exit completely.")

        recording = False
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video feed.")
                break

            frame = cv2.flip(frame, 1)  # Mirror frame

            # Extract ROI box (top right region)
            roi = frame[100:350, 300:550]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Otsu's thresholding for binary skin/background isolation
            _, thresholded = cv2.threshold(
                blur, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Morphological erosion/dilation to separate close fingers
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresholded = cv2.morphologyEx(
                thresholded, cv2.MORPH_OPEN, kernel, iterations=1
            )

            # Status UI
            color = (0, 255, 0) if recording else (0, 165, 255)
            status_text = "RECORDING..." if recording else "WAITING (Press 's')"
            cv2.rectangle(frame, (300, 100), (550, 350), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"Class: {category} [{count}/{SAMPLES_PER_CLASS}]",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Status: {status_text}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            if recording and count < SAMPLES_PER_CLASS:
                img_path = os.path.join(DATASET_PATH, category, f"{count}.jpg")
                cv2.imwrite(img_path, thresholded)
                count += 1
                if count >= SAMPLES_PER_CLASS:
                    recording = False
                    print(f"--> Successfully captured {category}!")
                    break  # Proceed to the next class

            cv2.imshow("Capture Pipeline (Main Feed)", frame)
            cv2.imshow("Processed ROI (Thresholded)", thresholded)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                recording = True
            elif key == ord("q"):
                print("Dataset capture aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()
    print("\nDataset recording completed for all classes!")


def load_data():
    """Loads recorded images, verifies counts, and formats arrays for training."""
    images, labels = [], []

    for idx, category in enumerate(CLASSES):
        folder = os.path.join(DATASET_PATH, category)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Missing folder: '{folder}'")

        file_list = os.listdir(folder)
        if len(file_list) == 0:
            raise ValueError(
                f"Folder '{folder}' is empty! Record images for all classes before training."
            )

        print(f"Loading '{category}': {len(file_list)} images found.")

        for img_name in file_list:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(resized)
                labels.append(idx)

    # Normalize pixels to range [0.0, 1.0] and shape to (N, 64, 64, 1)
    X = np.array(images, dtype="float32") / 255.0
    X = np.expand_dims(X, axis=-1)
    y = tf.keras.utils.to_categorical(
        np.array(labels), num_classes=len(CLASSES)
    )

    return X, y


def build_cnn_model():
    """Builds 2D CNN with modern spatial Data Augmentation layers."""
    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            # Integrated Spatial Augmentation
            layers.RandomRotation(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Dense Classifier
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # 1. Interactively Record Dataset
    create_dataset()

    # 2. Process & Verify Dataset
    print("\n--------------------------------------------------")
    print("Preparing images for model training...")
    try:
        X, y = load_data()
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        exit()

    # Split 80% Training / 20% Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Total dataset size: {len(X)} samples")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # 3. Train Architecture
    print("\n--------------------------------------------------")
    print("Initiating CNN Training...")
    model = build_cnn_model()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
    )

    # 4. Save Trained Model
    model.save(MODEL_SAVE_PATH)
    print("==================================================")
    print(f"Model saved successfully to '{MODEL_SAVE_PATH}'")
    print("Run 'python app.py' to launch real-time control!")
    print("==================================================")