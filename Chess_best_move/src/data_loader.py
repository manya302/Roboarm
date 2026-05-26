import os
import random
import cv2

def load_image(mode, dataset_path, fixed_image_name=None, uploaded_image_path=None):
    """
    Load image based on the specified mode
    """
    val_images_path = os.path.join(dataset_path, "valid", "images")
    
    if mode == "random":
        image_files = [f for f in os.listdir(val_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            raise ValueError(f"No images found in {val_images_path}")
        image_file = random.choice(image_files)
        image_path = os.path.join(val_images_path, image_file)
        print(f"\nSelected Random Image: {image_file}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image, image_path, image_file
    
    elif mode == "fixed":
        if not fixed_image_name:
            raise ValueError("fixed_image_name must be provided for fixed mode")
        image_path = os.path.join(val_images_path, fixed_image_name)
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")
        print(f"\nSelected Fixed Image: {fixed_image_name}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image, image_path, fixed_image_name
    
    elif mode == "upload":
        if not uploaded_image_path:
            raise ValueError("uploaded_image_path must be provided for upload mode")
        if not os.path.exists(uploaded_image_path):
            raise ValueError(f"Uploaded image not found: {uploaded_image_path}")
        print(f"\nUploaded Image: {uploaded_image_path}")
        image = cv2.imread(uploaded_image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {uploaded_image_path}")
        return image, uploaded_image_path, os.path.basename(uploaded_image_path)
    
    else:
        raise ValueError(f"Invalid MODE selected: {mode}")

def load_ground_truth(image_file, dataset_path):
    """
    Load ground truth labels for the given image
    """
    val_labels_path = os.path.join(dataset_path, "valid", "labels")
    label_file = image_file.rsplit(".", 1)[0] + ".txt"
    label_path = os.path.join(val_labels_path, label_file)
    
    if os.path.exists(label_path):
        return label_path
    return None