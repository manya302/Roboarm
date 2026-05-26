import cv2
import numpy as np

# Load DNN face detector
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

CONFIDENCE_THRESHOLD = 0.5


def detect_faces(frame):
    """
    Returns list of (x, y, w, h) face bounding boxes
    """
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            faces.append((x1, y1, x2 - x1, y2 - y1))

    return faces


def apply_face_mask(frame, faces):
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = 0
    return mask
