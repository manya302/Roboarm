import cv2
import json
import time
import numpy as np

from camera import get_camera
from face_detection import detect_faces, apply_face_mask
from colour_detection import process_colors
from hand_detection import detect_hands
from utils import draw_faces, group_objects, format_hands, detect_wave

cap = get_camera(0)

last_log_time = 0

# ---- ACTION STATE ----
current_action = "Waiting..."
action_start_time = 0
ACTION_DURATION = 1  # seconds
action_triggered = False
last_gesture = None

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---- FACE DETECTION ----
    faces = detect_faces(frame)
    face_mask = apply_face_mask(frame, faces)

    # ---- HAND DETECTION ----
    hands = detect_hands(frame)

    cv2.putText(frame, f"Hands Detected: {len(hands)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # ---- RESET ACTION AFTER DURATION ----
    if time.time() - action_start_time > ACTION_DURATION:
        if action_triggered:
            current_action = "Waiting..."
            action_triggered = False
            last_gesture = None

    # ---- HAND PROCESSING WITH ACTION MAPPING ----
    current_gesture = None

    for i, hand in enumerate(hands, start=1):
        x1, y1, x2, y2 = hand["box"]
        center_x = (x1 + x2) // 2

        # ---- WAVE DETECTION (only overwrite if wave is actually detected) ----
        if detect_wave(i, center_x):
            hand["gesture"] = "wave"

        # Read gesture AFTER potential wave override
        gesture = hand["gesture"]
        current_gesture = gesture

        # ---- ACTION MAPPING (only trigger if no active action) ----
        if not action_triggered:
            if gesture == "Fist":
                current_action = "Fist bump"
                action_start_time = time.time()
                action_triggered = True
                last_gesture = "Fist"

            elif gesture == "Open Hand":
                current_action = "Frozen! (hold to keep)"
                action_start_time = time.time()
                action_triggered = True
                last_gesture = "Open Hand"

            elif gesture == "wave":
                current_action = "Waves back"
                action_start_time = time.time()
                action_triggered = True
                last_gesture = "wave"
            elif gesture == "Thumbs Up":
                current_action = "Gives Thumbs Up"
                action_start_time = time.time()
                action_triggered = True
                last_gesture = "Thumbs Up"

        # ---- DRAW HAND ----
        label = f"{hand['side']} Hand - {hand['gesture']}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---- DEBUG OVERLAY ----
    cv2.putText(frame, f"Current Gesture: {current_gesture if current_gesture else 'None'}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Action Active: {action_triggered}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Apply face mask before color detection
    hsv = cv2.bitwise_and(hsv, hsv, mask=face_mask)

    # ---- INFORMATION ----
    objects = process_colors(frame, hsv)
    grouped_objects = group_objects(objects)
    hand_log = format_hands(hands)

    # ---- LOG DATA ----
    log_data = {
        "faces": len(faces),
        "hands": hand_log,
        "objects": grouped_objects,
        "current_action": current_action,
        "action_active": action_triggered
    }

    current_time = time.time()
    if current_time - last_log_time > 1:
        print("--------CAM LOGS--------\n")
        print(json.dumps(log_data, indent=2))
        last_log_time = current_time

    # ---- DRAW FACES ----
    draw_faces(frame, faces)

    # ---- MAIN WINDOW ----
    cv2.imshow("Face + Hand + Color Detection", frame)

    # ---- ACTION WINDOW ----
    action_frame = np.zeros((200, 500, 3), dtype=np.uint8)

    cv2.putText(action_frame, "BOT ACTION", (130, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(action_frame, current_action, (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if action_triggered:
        remaining_time = max(0, ACTION_DURATION - (time.time() - action_start_time))
        cv2.putText(action_frame, f"Remaining: {remaining_time:.1f}s", (50, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Robot Action", action_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()