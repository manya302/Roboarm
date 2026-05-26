#Handles reusable helper functions such as rendering face bounding boxes and labels on the video frame
import cv2
import numpy as np
from collections import defaultdict

def draw_faces(frame, faces):
    pnum = 1
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"Person {pnum}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        pnum += 1

def non_max_suppression(boxes, overlapThresh=0.4):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # sort by bottom-right y-coordinate
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:-1]]

        # delete picked index + overlapping ones
        idxs = np.delete(
            idxs,
            np.concatenate(([len(idxs) - 1],
                             np.where(overlap > overlapThresh)[0]))
        )

    return pick

def group_objects(objects):
    groups = defaultdict(int)

    for obj in objects:
        key = (obj["color"], obj["shape"])
        groups[key] += 1

    result = []
    obj_num = 1

    for (color, shape), count in groups.items():
        result.append(f"obj{obj_num}: {color}, {shape} ({count})")
        obj_num += 1

    return result

def format_hands(hands):
    left_id = 1
    right_id = 1
    result = []

    for h in hands:
        if h["side"] == "Left":
            label = f"left hand {left_id}: {h['gesture'].lower()}"
            left_id += 1
        else:
            label = f"right hand {right_id}: {h['gesture'].lower()}"
            right_id += 1

        result.append(label)

    return result

# utils.py

wave_history = {}

def detect_wave(hand_id, center_x):
    """
    Detects wave using horizontal motion over last few frames
    """

    if hand_id not in wave_history:
        wave_history[hand_id] = []

    history = wave_history[hand_id]
    history.append(center_x)

    if len(history) > 8:
        history.pop(0)

    if len(history) >= 6:
        movement = max(history) - min(history)

        # simple threshold
        if movement > 60:
            return True

    return False




