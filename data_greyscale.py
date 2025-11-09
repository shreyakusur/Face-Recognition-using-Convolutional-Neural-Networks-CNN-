import cv2
import os

# Input and output paths
input_folder = "frames"          # folder containing color images
output_gray = "gray_images"      # folder for greyscale images
output_binary = "binary_images"  # folder for binarized images

os.makedirs(output_gray, exist_ok=True)
os.makedirs(output_binary, exist_ok=True)

# Loop through all subfolders (each person)
for person_folder in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person_folder)
    if not os.path.isdir(person_path):
        continue

    # Create subfolders for outputs
    gray_person_path = os.path.join(output_gray, person_folder)
    binary_person_path = os.path.join(output_binary, person_folder)
    os.makedirs(gray_person_path, exist_ok=True)
    os.makedirs(binary_person_path, exist_ok=True)

    # Process each image
    for image_file in os.listdir(person_path):
        if not image_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(person_path, image_file)
        img = cv2.imread(img_path)

        # 1️⃣ Convert to Greyscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2️⃣ Apply Binarization (using a threshold value)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Save both versions
        cv2.imwrite(os.path.join(gray_person_path, image_file), gray)
        cv2.imwrite(os.path.join(binary_person_path, image_file), binary)

    print(f"✅ Processed {person_folder}")

print("\nAll images converted to greyscale and binarized successfully!")
