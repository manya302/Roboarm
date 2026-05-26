import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)


def detect_hands(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_data = []

    if results.multi_hand_landmarks:

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            lm = hand_landmarks.landmark

            # ---- LEFT / RIGHT ----
            handedness = results.multi_handedness[idx].classification[0].label

            fingers = []

            # Thumb (different logic for left & right)
            if handedness == "Right":
                fingers.append(1 if lm[4].x < lm[3].x else 0)
            else:
                fingers.append(1 if lm[4].x > lm[3].x else 0)

            # Other fingers (compare Y)
            fingers.append(1 if lm[8].y < lm[6].y else 0)   # Index
            fingers.append(1 if lm[12].y < lm[10].y else 0) # Middle
            fingers.append(1 if lm[16].y < lm[14].y else 0) # Ring
            fingers.append(1 if lm[20].y < lm[18].y else 0) # Pinky

            total_fingers = sum(fingers)

            # Bounding box
            h, w, _ = frame.shape
            x_coords = [p.x for p in lm]
            y_coords = [p.y for p in lm]

            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)

            # Gesture label
            if total_fingers == 0:
                gesture = "Fist"
            elif total_fingers == 5:
                gesture = "Open Hand"
            elif fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                gesture = "Thumbs Up"
            else:
                gesture = f"{total_fingers} Fingers"

            hand_data.append({
                "box": (x_min, y_min, x_max, y_max),
                "fingers": total_fingers,
                "gesture": gesture,
                "side": handedness
            })

    return hand_data
