import yaml
import os
import cv2
import numpy as np
from src.model_handler import load_model, run_detection
from src.fusion_logic import calculate_iou

# --- 1. Load Configuration ---
print("Loading configuration...")
with open('configs/main_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL_NAME = config['model_name']
DATA_PATH = config['data_path']
OUTPUT_PATH = config['output_path']
iou_threshold = 0.6

# --- 2. Load the AI Model (only once) ---
model = load_model(MODEL_NAME)

# --- 3. Find Image Pairs in the Data Folder ---
print(f"--- Searching for image pairs in '{DATA_PATH}' folder... ---")
# Find all files ending with '_eo.jpg'
eo_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith('_eo.jpg')])
image_pairs = []
for eo_file in eo_files:
    # Check if a corresponding '_ir.jpg' file exists
    ir_file = eo_file.replace('_eo.jpg', '_ir.jpg')
    if os.path.exists(os.path.join(DATA_PATH, ir_file)):
        image_pairs.append((eo_file, ir_file))
        print(f"Found pair: ({eo_file}, {ir_file})")

# --- 4. Create Results Directory ---
os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"Results will be saved in '{OUTPUT_PATH}' folder.")

# --- 5. Process Each Image Pair in a Loop ---
for eo_filename, ir_filename in image_pairs:
    print(f"\n--- Processing pair: {eo_filename}, {ir_filename} ---")
    eo_image_path = os.path.join(DATA_PATH, eo_filename)
    ir_image_path = os.path.join(DATA_PATH, ir_filename)

    # Run detection on both images
    results_eo = run_detection(model, eo_image_path)
    results_ir = run_detection(model, ir_image_path)

    # Get bounding boxes
    boxes_eo = results_eo[0].boxes.xyxy.cpu().numpy()
    boxes_ir = results_ir[0].boxes.xyxy.cpu().numpy()
    verified_indices = []

    # Fusion Logic to find matches
    for i, box_eo in enumerate(boxes_eo):
        for j, box_ir in enumerate(boxes_ir):
            iou = calculate_iou(box_eo, box_ir)
            if iou > iou_threshold:
                verified_indices.append(i)
                break

    # Prepare images for display
    annotated_eo = results_eo[0].plot()
    annotated_ir = results_ir[0].plot()

    # Re-draw verified boxes in red
    for i in verified_indices:
        box = boxes_eo[i]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_eo, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)

    # Resize and combine images
    target_height = 500
    h1, w1, _ = annotated_eo.shape
    h2, w2, _ = annotated_ir.shape
    scale1 = target_height / h1
    scale2 = target_height / h2
    resized_eo = cv2.resize(annotated_eo, (int(w1 * scale1), target_height))
    resized_ir = cv2.resize(annotated_ir, (int(w2 * scale2), target_height))
    side_by_side_image = np.concatenate((resized_eo, resized_ir), axis=1)

    # --- 6. Save the Output Image ---
    output_filename = eo_filename.replace('_eo.jpeg', '_fused_result.jpeg')
    output_filepath = os.path.join(OUTPUT_PATH, output_filename)
    cv2.imwrite(output_filepath, side_by_side_image)
    print(f"✅ Fused result saved to: {output_filepath}")

print("\nBatch processing complete.")