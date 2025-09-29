import yaml
import os
import cv2
import numpy as np
from src.model_handler import load_model, run_detection
from src.fusion_logic import calculate_iou # <-- Import our new function

# --- 1. Load Configuration ---
print("Loading configuration...")
with open('configs/main_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL_NAME = config['model_name']
DATA_PATH = config['data_path']

# --- 2. Define Image Paths ---
EO_IMAGE_FILENAME = 'test_eo.jpg'
IR_IMAGE_FILENAME = 'test_ir.jpg'
eo_image_path = os.path.join(DATA_PATH, EO_IMAGE_FILENAME)
ir_image_path = os.path.join(DATA_PATH, IR_IMAGE_FILENAME)

# --- 3. Load the AI Model ---
model = load_model(MODEL_NAME)

# --- 4. Run Detection on Both Images ---
print("--- Running detection on Visual (EO) image... ---")
results_eo = run_detection(model, eo_image_path)
print("--- Running detection on Thermal (IR) image... ---")
results_ir = run_detection(model, ir_image_path)

# --- 5. Fusion Logic: Find Matching Detections ---
print("--- Fusing detection results... ---")
iou_threshold = 0.6  # Set the required overlap percentage (60%)
verified_indices = []

# Get the actual bounding box coordinates from the results
boxes_eo = results_eo[0].boxes.xyxy.cpu().numpy()
boxes_ir = results_ir[0].boxes.xyxy.cpu().numpy()

# Compare every EO box with every IR box
for i, box_eo in enumerate(boxes_eo):
    for j, box_ir in enumerate(boxes_ir):
        iou = calculate_iou(box_eo, box_ir)
        if iou > iou_threshold:
            print(f"✅ Verified Threat! Match found with IoU: {iou:.2f}")
            verified_indices.append(i)
            break # Stop searching for this EO box once a match is found

# --- 6. Prepare Images for Display ---
annotated_eo = results_eo[0].plot() # This draws the default blue boxes
annotated_ir = results_ir[0].plot()

# Re-draw the verified boxes in RED on the EO image
for i in verified_indices:
    box = boxes_eo[i]
    x1, y1, x2, y2 = map(int, box) # Convert coordinates to integers
    # Draw a thick, bright red rectangle
    cv2.rectangle(annotated_eo, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)

# Resize and combine images for side-by-side view
target_height = 500
h1, w1, _ = annotated_eo.shape
h2, w2, _ = annotated_ir.shape
scale1 = target_height / h1
scale2 = target_height / h2
resized_eo = cv2.resize(annotated_eo, (int(w1 * scale1), target_height))
resized_ir = cv2.resize(annotated_ir, (int(w2 * scale2), target_height))
side_by_side_image = np.concatenate((resized_eo, resized_ir), axis=1)

# --- 7. Display the Final Result ---
print("Displaying fused results. Press any key to exit.")
cv2.imshow("Fused EO vs IR Detection Results", side_by_side_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Script finished.")