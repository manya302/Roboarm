# Determines the geometric shape of a detected contour
import cv2

def detect_shape(cnt):
    shape = "Unknown"

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    if len(approx) == 3:
        shape = "Triangle"

    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"

    elif len(approx) > 6:

        # Need at least 5 points to fit ellipse
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)

            axis_ratio = major_axis / minor_axis

            # If axes are almost equal → Circle
            if 0.9 <= axis_ratio <= 1.1:
                shape = "Circle"
            else:
                shape = "Ellipse"
        else:
            shape = "Circle"

    return shape
