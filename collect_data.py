import cv2
import mediapipe as mp
import csv
import os

# MediaPipe setup
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

# Gestures
gestures = [
    "fist",
    "open_palm",
    "thumbs_up",
    "thumbs_down",
    "peace",
    "pointing",
    "ok"
]

# Create CSV file
csv_file = "gesture_data.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["label"] +
            [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
        )

# Choose gesture
print("\nAvailable gestures:")
for i, gesture in enumerate(gestures):
    print(f"{i + 1}. {gesture}")

choice = int(input("\nEnter gesture number: "))

if choice < 1 or choice > len(gestures):
    print("Invalid choice.")
    landmarker.close()
    exit()

label = gestures[choice - 1]

print(f"\nCollecting data for: {label}")
print("Show your hand to the camera.")
print("Press SPACE to start collecting.")
print("Press Q to quit.")

cap = cv2.VideoCapture(0)

collecting = False
count = 0

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

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Draw landmarks
        for landmark in hand:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        if collecting:
            row = [label]

            for landmark in hand:
                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            with open(csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

            count += 1

    status = "COLLECTING" if collecting else "READY"

    cv2.putText(
        frame,
        f"Gesture: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"{status} | Samples: {count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Gesture Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        collecting = True
        print("Collection started!")

    elif key == ord("q"):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print(f"\nCollected {count} samples for {label}.")
print(f"Saved to {csv_file}")