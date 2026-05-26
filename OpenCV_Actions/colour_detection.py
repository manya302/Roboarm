# Identifies colored regions in the frame and detects shapes within those regions
import cv2
import numpy as np
from shape_detection import detect_shape
from utils import non_max_suppression


def get_color_data(hsv):
    return [
        ("Red",
         cv2.inRange(hsv, (0,150,120), (8,255,255)) +
         cv2.inRange(hsv, (172,150,120), (180,255,255)),
         (0, 0, 255)),

        ("Orange", cv2.inRange(hsv, (12,160,140), (20,255,255)), (0,165,255)),
        ("Yellow", cv2.inRange(hsv, (26,120,120), (35,255,255)), (0,255,255)),
        ("Green", cv2.inRange(hsv, (40,70,70), (80,255,255)), (0,255,0)),
        ("Blue", cv2.inRange(hsv, (100,150,50), (140,255,255)), (255,0,0)),
        ("Purple", cv2.inRange(hsv, (140,100,50), (165,255,255)), (255,0,255)),
        ("Black", cv2.inRange(hsv, (0,0,0), (180,255,40)), (50,50,50)),
        ("White", cv2.inRange(hsv, (0,0,200), (180,30,255)), (255,255,255))
    ]


def process_colors(frame, hsv):
    kernel = np.ones((5,5), np.uint8)

    boxes = []
    labels = []
    colors = []
    detections = []   # <-- NEW

    for color_name, mask, box_color in get_color_data(hsv):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) > 800:
                x, y, w, h = cv2.boundingRect(cnt)
                shape = detect_shape(cnt)

                boxes.append([x, y, x + w, y + h])
                labels.append(f"{color_name}, {shape}")
                colors.append(box_color)

                detections.append({
                    "color": color_name.lower(),
                    "shape": shape.lower()
                })

    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes)
    keep = non_max_suppression(boxes_np, overlapThresh=0.4)

    final_objects = []

    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], 2)
        cv2.putText(frame, labels[i], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

        final_objects.append(detections[i])

    return final_objects
