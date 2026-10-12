import cv2
import mediapipe as mp
import pyautogui
import time

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

start_x = None
start_time = None
swipe_lock = False

print("===================================")
print("          AI AIR SWIPE")
print("===================================")
print("Swipe LEFT  = Browser Back")
print("Swipe RIGHT = Browser Forward")
print("Press Q = Quit")
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

        x = index_finger.x
        y = index_finger.y

        current_time = time.time()

        if start_x is None:
            start_x = x
            start_time = current_time

        elapsed = current_time - start_time
        distance = x - start_x

        if elapsed < 0.7 and not swipe_lock:

            if distance > 0.30:

                pyautogui.hotkey("alt", "right")
                print("SWIPE RIGHT")

                swipe_lock = True

            elif distance < -0.30:

                pyautogui.hotkey("alt", "left")
                print("SWIPE LEFT")

                swipe_lock = True

        if elapsed > 0.8:

            start_x = x
            start_time = current_time
            swipe_lock = False

        px = int(x * frame.shape[1])
        py = int(y * frame.shape[0])

        cv2.circle(
            frame,
            (px, py),
            12,
            (0, 255, 0),
            -1
        )

    cv2.putText(
        frame,
        "AI AIR SWIPE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "LEFT = BACK     RIGHT = FORWARD",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AI Air Swipe",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print("AI Air Swipe stopped.")