import json
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# !!! IMPORTANT: UPDATE THESE PATHS !!!
# Path to the root of your downloaded FLIR ADAS dataset
FLIR_DATASET_PATH = "D:/FLIR_ADAS_v2" 
# Path to the main annotation file
ANNOTATION_FILE_PATH = "D:/FLIR_ADAS_v2/images_thermal_train/coco.json" 
# Where you want to save the new YOLO-formatted dataset
YOLO_DATASET_PATH = "D:/FLIR_YOLO_Dataset" 

# --- SCRIPT ---

def convert_coco_to_yolo():
    """
    Converts COCO annotation format to YOLOv8 format.
    Creates the necessary folder structure and label files.
    """
    # Load the COCO JSON annotation file
    try:
        with open(ANNOTATION_FILE_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Annotation file not found at '{ANNOTATION_FILE_PATH}'")
        return

    # Create the main YOLO dataset directory
    os.makedirs(YOLO_DATASET_PATH, exist_ok=True)

    # A map from category ID to a simple 0-indexed integer
    category_map = {cat['id']: i for i, cat in enumerate(data['categories'])}
    print(f"Found categories: {[cat['name'] for cat in data['categories']]}")

    # Process each image in the annotation file
    print("Processing annotations and creating YOLO label files...")
    for image in tqdm(data['images']):
        image_id = image['id']
        file_name = image['file_name']
        img_width = image['width']
        img_height = image['height']

        # Determine if this image is for training or validation
        # This is a simple split, you can make it more robust later
        subset = 'train' if 'train' in file_name else 'val'
        
        # Create directories
        yolo_images_path = os.path.join(YOLO_DATASET_PATH, subset, 'images')
        yolo_labels_path = os.path.join(YOLO_DATASET_PATH, subset, 'labels')
        os.makedirs(yolo_images_path, exist_ok=True)
        os.makedirs(yolo_labels_path, exist_ok=True)

        # Copy the image file to the new location
        original_image_path = os.path.join(FLIR_DATASET_PATH, file_name)
        new_image_path = os.path.join(yolo_images_path, os.path.basename(file_name))
        if os.path.exists(original_image_path):
            os.replace(original_image_path, new_image_path)

        # Create the corresponding label file
        label_file_path = os.path.join(yolo_labels_path, os.path.splitext(os.path.basename(file_name))[0] + '.txt')
        with open(label_file_path, 'w') as lf:
            # Find all annotations for this image
            annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
            for ann in annotations:
                category_id = category_map[ann['category_id']]
                
                # COCO format is [x_min, y_min, width, height]
                x_min, y_min, width, height = ann['bbox']

                # YOLO format is [class_id, x_center_norm, y_center_norm, width_norm, height_norm]
                x_center = x_min + width / 2
                y_center = y_min + height / 2

                # Normalize coordinates
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = width / img_width
                height_norm = height / img_height

                lf.write(f"{category_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

    print("\nDataset conversion complete!")
    print(f"YOLO-formatted dataset is ready at: {YOLO_DATASET_PATH}")


if __name__ == "__main__":
    # Before running, make sure you have tqdm installed: pip install tqdm
    convert_coco_to_yolo()