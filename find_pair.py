import os

# !!! IMPORTANT !!!
# Change this path to the exact location of your downloaded and unzipped FLIR dataset folder.
# Use forward slashes '/' in your path.
DATASET_PATH = "D:/FLIR_ADAS_v2"

# --- Do not change the code below this line ---

# Define the paths to the two folders we want to compare
rgb_folder = os.path.join(DATASET_PATH, "images_rgb_train")
thermal_folder = os.path.join(DATASET_PATH, "images_thermal_train")

print(f"Searching for a matching pair...")

try:
    # Get a set of all filenames in each folder for fast comparison
    rgb_files = set(os.listdir(rgb_folder))
    thermal_files = set(os.listdir(thermal_folder))

    # Find the files that exist in both sets
    common_files = rgb_files.intersection(thermal_files)

    if common_files:
        # Get just one example from the set of common files
        match = common_files.pop()
        print("\n✅ --- MATCH FOUND! --- ✅")
        print(f"Use this filename: {match}")
        print("\nYou can now find this file in both the 'images_rgb_train' and 'images_thermal_train' folders.")
    else:
        print("\n❌ --- NO MATCH FOUND --- ❌")
        print("Could not find any files with the same name in both folders.")

except FileNotFoundError:
    print("\n❌ --- FOLDER NOT FOUND ERROR --- ❌")
    print("Could not find the dataset folders. Please double-check the DATASET_PATH variable in your script.")