import yaml
import os
import cv2
import numpy as np # New library for combining images
from src.model_handler import load_model, run_detection

# --- 1. Load the Configuration File ---
print("Loading configuration...")
with open('configs/main_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# --- 2. Get Settings from the Config ---
MODEL_NAME = config['model_name']
DATA_PATH = config['data_path']

# Define the filenames for our new EO/IR pair
EO_IMAGE_FILENAME = 'test_eo.jpg'
IR_IMAGE_FILENAME = 'test_ir.jpg'

eo_image_path = os.path.join(DATA_PATH, EO_IMAGE_FILENAME)
ir_image_path = os.path.join(DATA_PATH, IR_IMAGE_FILENAME)

# --- 3. Load the AI Model ---
model = load_model(MODEL_NAME)

# --- 4. Run Detection on BOTH Images ---
print("--- Running detection on Visual (EO) image... ---")
results_eo = run_detection(model, eo_image_path)

print("--- Running detection on Thermal (IR) image... ---")
results_ir = run_detection(model, ir_image_path)

# --- 5. Prepare Images for Side-by-Side Display ---
annotated_eo = results_eo[0].plot()
annotated_ir = results_ir[0].plot()

# To combine them, we'll resize them to the same height
target_height = 500 # You can adjust this value
h1, w1, _ = annotated_eo.shape
h2, w2, _ = annotated_ir.shape

scale1 = target_height / h1
scale2 = target_height / h2

resized_eo = cv2.resize(annotated_eo, (int(w1 * scale1), target_height))
resized_ir = cv2.resize(annotated_ir, (int(w2 * scale2), target_height))

# Use numpy to concatenate the images horizontally
side_by_side_image = np.concatenate((resized_eo, resized_ir), axis=1)

# --- 6. Display the Final Result ---
print("Displaying side-by-side results. Press any key to exit.")
cv2.imshow("EO vs IR Detection Results", side_by_side_image)
cv2.waitKey(0) # Wait for a key press before closing the window
cv2.destroyAllWindows()

print("Script finished.")