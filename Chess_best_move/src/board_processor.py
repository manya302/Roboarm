import cv2
import numpy as np

def detect_board_contour(image):
    """
    Detect the chess board contour in the image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in image")
    
    board_contour = max(contours, key=cv2.contourArea)
    return board_contour

def get_perspective_transform(board_contour, board_size):
    """
    Get perspective transform matrix for the board
    """
    hull = cv2.convexHull(board_contour)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    
    pts = approx.reshape(-1, 2)
    
    # If more than 4 points, take extreme corners
    if len(pts) > 4:
        rect = cv2.minAreaRect(board_contour)
        box = cv2.boxPoints(rect)
        pts = box.astype(np.int32)
    
    # Order points
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    dst = np.array([
        [0, 0],
        [board_size - 1, 0],
        [board_size - 1, board_size - 1],
        [0, board_size - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return M

def warp_board(image, M, board_size):
    """
    Apply perspective transform to get top-down view of the board
    """
    warped = cv2.warpPerspective(image, M, (board_size, board_size))
    return warped

def draw_grid(warped, board_size):
    """
    Draw grid on the warped board
    """
    grid = warped.copy()
    square_size = board_size // 8
    
    for i in range(9):
        cv2.line(grid, (i * square_size, 0), (i * square_size, board_size), (255, 0, 0), 1)
        cv2.line(grid, (0, i * square_size), (board_size, i * square_size), (255, 0, 0), 1)
    
    return grid, square_size