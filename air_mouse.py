import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

screen_width, screen_height = pyautogui.size()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

pinching = False
last_pinch_time = 0
double_click_window = 0.4

print("===================================")
print("        AI AIR MOUSE")
print("===================================")
print("Index finger  = Move cursor")
print("Pinch         = Click")
print("Quick pinch twice = Double click")
print("Press Q       = Quit")
print("===================================")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        index_finger = hand[8]
        thumb = hand[4]

        screen_x = np.interp(
            index_finger.x,
            [0.1, 0.9],
            [0, screen_width]
        )

        screen_y = np.interp(
            index_finger.y,
            [0.1, 0.9],
            [0, screen_height]
        )

        pyautogui.moveTo(
            screen_x,
            screen_y,
            duration=0.05
        )

        distance = np.sqrt(
            (index_finger.x - thumb.x) ** 2 +
            (index_finger.y - thumb.y) ** 2
        )

        current_time = time.time()

        if distance < 0.05 and not pinching:

            pinching = True

            if current_time - last_pinch_time < double_click_window:

                pyautogui.doubleClick(
                    interval=0.15
                )

                print("DOUBLE CLICK")

                last_pinch_time = 0

            else:

                pyautogui.click()

                print("CLICK")

                last_pinch_time = current_time

        if distance > 0.08:

            pinching = False

        x = int(
            index_finger.x * frame.shape[1]
        )

        y = int(
            index_finger.y * frame.shape[0]
        )

        cv2.circle(
            frame,
            (x, y),
            10,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            "AIR MOUSE ACTIVE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        if distance < 0.05:

            cv2.putText(
                frame,
                "PINCH",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    cv2.imshow(
        "AI Air Mouse",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print("Air mouse stopped.")