import yaml
import os
import cv2
from src.model_handler import load_model, run_detection

# --- 1. Load the Configuration File ---
print("Loading configuration...")
with open('configs/main_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# --- 2. Get Settings from the Config ---
MODEL_NAME = config['model_name']
DATA_PATH = config['data_path']

# --- THIS IS A PLACEHOLDER FOR OUR FIRST TEST ---
# !!! IMPORTANT: Change this to the exact name of the image you put in the 'data' folder.
TEST_IMAGE_FILENAME ='street_photo.jpg'

image_path = os.path.join(DATA_PATH, TEST_IMAGE_FILENAME)

# --- 3. Load the AI Model ---
model = load_model(MODEL_NAME)

# --- 4. Run Detection on the Image ---
results = run_detection(model, image_path)

# --- 5. Display the Results ---
# The results object contains the annotated image in a numpy array format
annotated_image = results[0].plot()

print("Displaying detection results. Press any key to exit.")
cv2.imshow("Detection Results", annotated_image)
cv2.waitKey(0) # Wait for a key press before closing the window
cv2.destroyAllWindows()

print("Script finished.")