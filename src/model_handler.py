from ultralytics import YOLO

def load_model(model_name):
    """
    Loads the YOLOv8 model from the specified file.
    """
    model = YOLO(model_name)
    print(f"AI Model '{model_name}' loaded successfully.")
    return model

def run_detection(model, image_path):
    """
    Runs object detection on a single image.
    """
    results = model(image_path)
    print(f"Detection complete for image: {image_path}")
    return results