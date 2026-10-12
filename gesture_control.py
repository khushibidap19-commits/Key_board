import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyautogui
import time

# Load trained model
model = load_model("gesture_model.h5")
labels = np.load("label_encoder.npy", allow_pickle=True)

# MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

print("===================================")
print(" AI HAND GESTURE CONTROL STARTED")
print("===================================")
print("Press Q to quit.")

last_gesture = ""
last_action_time = 0
cooldown = 1.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not access camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    gesture = "No hand"
    confidence = 0.0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Extract 21 landmarks
        features = []

        for landmark in hand:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # Convert to NumPy array
        features = np.array(features).reshape(1, -1)

        # Predict gesture
        prediction = model.predict(features, verbose=0)

        predicted_index = np.argmax(prediction)
        confidence = float(prediction[0][predicted_index])

        gesture = labels[predicted_index]

        # Draw landmarks
        for landmark in hand:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Perform action only with high confidence
        current_time = time.time()

        if confidence > 0.90 and current_time - last_action_time > cooldown:

            if gesture == "fist":
                pyautogui.press("space")

            elif gesture == "open_palm":
                pyautogui.press("space")

            elif gesture == "thumbs_up":
                pyautogui.press("volumeup")

            elif gesture == "thumbs_down":
                pyautogui.press("volumedown")

            elif gesture == "peace":
                pyautogui.press("right")

            elif gesture == "pointing":
                pyautogui.press("left")

            elif gesture == "ok":
                pyautogui.press("enter")

            last_action_time = current_time

    # Display gesture
    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("AI Hand Gesture Control stopped.")