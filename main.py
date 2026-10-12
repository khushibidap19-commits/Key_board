import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
import time
import math

MODEL_PATH = "gesture_model.h5"
LABEL_PATH = "label_encoder.npy"
HAND_MODEL = "hand_landmarker.task"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = np.load(LABEL_PATH, allow_pickle=True)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mode = "MOUSE"

pinching = False
last_click_time = 0
last_swipe_time = 0
swipe_start_x = None
swipe_start_time = None

keyboard_rows = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M"],
    ["SPACE","BACK","ENTER"]
]

keyboard_key = None
last_key_time = 0

def distance(a, b):
    return math.sqrt(
        (a.x-b.x)**2 +
        (a.y-b.y)**2 +
        (a.z-b.z)**2
    )

def fingers_up(hand):
    return (
        hand[8].y < hand[6].y,
        hand[12].y < hand[10].y,
        hand[16].y < hand[14].y,
        hand[20].y < hand[18].y
    )

def draw_keyboard(frame, cursor_x, cursor_y):
    global keyboard_key

    h, w = frame.shape[:2]

    start_x = 40
    start_y = 300
    key_w = 75
    key_h = 55
    gap = 8

    keyboard_key = None

    for r, row in enumerate(keyboard_rows):

        if r == 0:
            row_x = start_x
        elif r == 1:
            row_x = start_x + 35
        elif r == 2:
            row_x = start_x + 75
        else:
            row_x = start_x + 170

        for c, key in enumerate(row):

            if r == 3:
                if key == "SPACE":
                    kw = 230
                elif key == "BACK":
                    kw = 130
                else:
                    kw = 130
            else:
                kw = key_w

            x1 = row_x + c * (key_w + gap)
            y1 = start_y + r * (key_h + gap)
            x2 = x1 + kw
            y2 = y1 + key_h

            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                keyboard_key = key
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    -1
                )
                cv2.putText(
                    frame,
                    key,
                    (x1 + 10, y1 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    2
                )
            else:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    key,
                    (x1 + 10, y1 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

    return frame

with HandLandmarker.create_from_options(options) as landmarker:

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(frame_number * 33.33)
        frame_number += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        cursor_x = 0
        cursor_y = 0

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            points = []

            for p in hand:
                points.extend([p.x, p.y, p.z])

            input_data = np.array(points).reshape(1, -1)

            prediction = model.predict(
                input_data,
                verbose=0
            )

            index = np.argmax(prediction)
            confidence = float(prediction[0][index])

            gesture = str(labels[index])

            index_tip = hand[8]
            thumb_tip = hand[4]
            wrist = hand[0]

            cursor_x = int(index_tip.x * w)
            cursor_y = int(index_tip.y * h)

            pinch_distance = distance(
                index_tip,
                thumb_tip
            )

            if mode == "MOUSE":

                target_x = np.interp(
                    index_tip.x,
                    [0.08, 0.92],
                    [0, screen_w]
                )

                target_y = np.interp(
                    index_tip.y,
                    [0.08, 0.92],
                    [0, screen_h]
                )

                target_x = max(0, min(screen_w - 1, target_x))
                target_y = max(0, min(screen_h - 1, target_y))

                try:
                    pyautogui.moveTo(
                        int(target_x),
                        int(target_y),
                        duration=0.01
                    )
                except:
                    pass

                current_time = time.time()

                if pinch_distance < 0.055:

                    if not pinching:

                        if current_time - last_click_time < 0.45:
                            pyautogui.doubleClick(
                                interval=0.12
                            )
                            last_click_time = 0
                        else:
                            pyautogui.click()
                            last_click_time = current_time

                        pinching = True

                elif pinch_distance > 0.085:

                    pinching = False

                if swipe_start_x is None:

                    swipe_start_x = index_tip.x
                    swipe_start_time = current_time

                else:

                    dx = index_tip.x - swipe_start_x
                    elapsed = current_time - swipe_start_time

                    if elapsed < 0.7 and abs(dx) > 0.30:

                        if current_time - last_swipe_time > 1.0:

                            if dx > 0:
                                pyautogui.hotkey(
                                    "alt",
                                    "right"
                                )
                            else:
                                pyautogui.hotkey(
                                    "alt",
                                    "left"
                                )

                            last_swipe_time = current_time

                        swipe_start_x = index_tip.x
                        swipe_start_time = current_time

                    elif elapsed >= 0.7:

                        swipe_start_x = index_tip.x
                        swipe_start_time = current_time

                if gesture == "open_palm":

                    scroll_amount = int(
                        (0.5 - wrist.y) * 20
                    )

                    if abs(scroll_amount) > 2:

                        pyautogui.scroll(
                            scroll_amount
                        )

                cv2.putText(
                    frame,
                    "MOUSE MODE",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            if mode == "KEYBOARD":

                frame = draw_keyboard(
                    frame,
                    cursor_x,
                    cursor_y
                )

                current_time = time.time()

                if pinch_distance < 0.055:

                    if not pinching and keyboard_key:

                        if current_time - last_key_time > 0.35:

                            if keyboard_key == "SPACE":

                                pyautogui.press("space")

                            elif keyboard_key == "BACK":

                                pyautogui.press("backspace")

                            elif keyboard_key == "ENTER":

                                pyautogui.press("enter")

                            else:

                                pyautogui.write(
                                    keyboard_key.lower()
                                )

                            last_key_time = current_time

                        pinching = True

                elif pinch_distance > 0.085:

                    pinching = False

                cv2.putText(
                    frame,
                    "KEYBOARD MODE",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                "Gesture: " + gesture,
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Confidence: " + str(round(confidence * 100, 1)) + "%",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "M = Mouse   K = Keyboard   Q = Quit",
                (30, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            for landmark in hand:

                px = int(landmark.x * w)
                py = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (px, py),
                    5,
                    (0, 255, 0),
                    -1
                )

        cv2.imshow(
            "AI Gesture Computer Control",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("m"):
            mode = "MOUSE"
            pinching = False
            swipe_start_x = None

        if key == ord("k"):
            mode = "KEYBOARD"
            pinching = False
            swipe_start_x = None

cap.release()
cv2.destroyAllWindows()
