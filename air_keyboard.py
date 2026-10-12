import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import tkinter as tk
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

keyboard = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M"],
    ["SPACE", "BACKSPACE", "ENTER"]
]

key_x = {
    "Q": 100, "W": 180, "E": 260, "R": 340, "T": 420,
    "Y": 500, "U": 580, "I": 660, "O": 740, "P": 820,
    "A": 140, "S": 220, "D": 300, "F": 380, "G": 460,
    "H": 540, "J": 620, "K": 700, "L": 780,
    "Z": 260, "X": 340, "C": 420, "V": 500,
    "B": 580, "N": 660, "M": 740
}

key_y = {
    "Q": 100, "W": 100, "E": 100, "R": 100, "T": 100,
    "Y": 100, "U": 100, "I": 100, "O": 100, "P": 100,
    "A": 180, "S": 180, "D": 180, "F": 180, "G": 180,
    "H": 180, "J": 180, "K": 180, "L": 180,
    "Z": 260, "X": 260, "C": 260, "V": 260,
    "B": 260, "N": 260, "M": 260
}

last_key = ""
pinching = False
last_press_time = 0
press_cooldown = 0.5

print("===================================")
print("       AI AIR KEYBOARD")
print("===================================")
print("Move index finger over a key.")
print("Pinch thumb + index finger to type.")
print("Press Q to quit.")
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

    selected_key = ""

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        index_finger = hand[8]
        thumb = hand[4]

        x = int(index_finger.x * frame.shape[1])
        y = int(index_finger.y * frame.shape[0])

        keyboard_width = frame.shape[1]
        keyboard_height = frame.shape[0]

        for key, kx in key_x.items():

            ky = key_y[key]

            real_x = int(
                kx * keyboard_width / 900
            )

            real_y = int(
                ky * keyboard_height / 400
            )

            if (
                real_x - 35 < x < real_x + 35
                and
                real_y - 30 < y < real_y + 30
            ):
                selected_key = key
                break

        distance = np.sqrt(
            (index_finger.x - thumb.x) ** 2 +
            (index_finger.y - thumb.y) ** 2
        )

        current_time = time.time()

        if (
            distance < 0.05
            and not pinching
            and selected_key != ""
            and current_time - last_press_time > press_cooldown
        ):

            pinching = True
            last_press_time = current_time

            if selected_key == "SPACE":
                pyautogui.press("space")
                print("SPACE")

            elif selected_key == "BACKSPACE":
                pyautogui.press("backspace")
                print("BACKSPACE")

            elif selected_key == "ENTER":
                pyautogui.press("enter")
                print("ENTER")

            else:
                pyautogui.write(selected_key.lower())
                print("TYPED:", selected_key)

        if distance > 0.08:
            pinching = False

        cv2.circle(
            frame,
            (x, y),
            10,
            (0, 255, 0),
            -1
        )

    start_x = 40
    start_y = 80
    key_width = 70
    key_height = 55
    gap = 8

    for row_index, row in enumerate(keyboard):

        for col_index, key in enumerate(row):

            if row_index == 0:
                x1 = start_x + col_index * (key_width + gap)

            elif row_index == 1:
                x1 = start_x + 35 + col_index * (key_width + gap)

            elif row_index == 2:
                x1 = start_x + 105 + col_index * (key_width + gap)

            else:
                x1 = start_x + col_index * 190

            y1 = start_y + row_index * 70

            if key == "SPACE":
                width = 170
            elif key == "BACKSPACE":
                width = 170
            elif key == "ENTER":
                width = 170
            else:
                width = key_width

            if key == selected_key:
                thickness = 4
            else:
                thickness = 2

            cv2.rectangle(
                frame,
                (x1, y1),
                (x1 + width, y1 + key_height),
                (255, 255, 255),
                thickness
            )

            cv2.putText(
                frame,
                key,
                (x1 + 12, y1 + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    cv2.putText(
        frame,
        "AI AIR KEYBOARD",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Point + Pinch = Type",
        (20, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AI Air Keyboard",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print("AI Air Keyboard stopped.")