import cv2
import numpy as np
from ultralytics import YOLO

def load_model(model_path):
    """
    Load YOLO model
    """
    model = YOLO(model_path)
    return model

def detect_pieces(model, image_path, confidence_threshold, M, board_size, square_size, files, ranks):
    """
    Detect chess pieces and map them to board squares
    """
    results = model.predict(source=image_path, conf=confidence_threshold, save=False)
    
    square_predictions = {}
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        
        x1, y1, x2, y2 = box.xyxy[0]
        x_center = float((x1 + x2) / 2)
        y_bottom = float(y2)
        
        point = np.array([[[x_center, y_bottom]]], dtype="float32")
        warped_point = cv2.perspectiveTransform(point, M)
        
        wx = warped_point[0][0][0]
        wy = warped_point[0][0][1]
        
        # Check if point is within board
        if wx < 0 or wx >= board_size or wy < 0 or wy >= board_size:
            print(f"Ignored {class_name} (outside board)")
            continue
        
        col = int(wx / square_size)
        row = int(wy / square_size)
        
        # Ensure within bounds
        col = min(max(col, 0), 7)
        row = min(max(row, 0), 7)
        
        square = files[row] + ranks[col]
        
        # Keep highest confidence prediction for each square
        if square not in square_predictions or confidence > square_predictions[square][1]:
            square_predictions[square] = (class_name, confidence)
    
    # Convert to list of (class_name, square)
    predicted_data = [(class_name, square) for square, (class_name, _) in square_predictions.items()]
    
    return results, predicted_data