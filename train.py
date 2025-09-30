from ultralytics import YOLO

# --- CONFIGURATION ---

# Path to the dataset configuration YAML file
DATASET_YAML_PATH = 'flir_dataset.yaml'

# Choose the base model to start from (e.g., yolov8n.pt for nano)
BASE_MODEL = 'yolov8n.pt'

# Number of training epochs (how many times to go through the dataset)
EPOCHS = 1 

# Image size
IMAGE_SIZE = 640


# --- TRAINING SCRIPT ---
if __name__ == '__main__':
    print("Loading base model...")
    # Load a pre-trained model
    model = YOLO(BASE_MODEL)

    print("Starting training...")
    # Train the model on your custom dataset
    results = model.train(
        data=DATASET_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project='runs', # Directory to save results
        name='train_results' # Sub-directory name for this training run
    )

    print("Training complete.")
    print("Your trained model and results are saved in the 'runs/train_results' folder.")