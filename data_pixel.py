import cv2
import os
import numpy as np

# Input: resized images from Part 3
input_folder = "resized_faces"
output_folder = "normalized_faces"
os.makedirs(output_folder, exist_ok=True)

# Loop through all 5 persons' folders
for person_folder in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person_folder)
    if not os.path.isdir(person_path):
        continue

    save_path = os.path.join(output_folder, person_folder)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n🔹 Normalizing images for: {person_folder}")

    for img_file in os.listdir(person_path):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # optional, keeps consistent color order

        # Normalize pixel values (0–255 → 0–1)
        normalized_img = img / 255.0

        # Convert back to 0–255 for saving (optional)
        normalized_save = (normalized_img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, img_file), cv2.cvtColor(normalized_save, cv2.COLOR_RGB2BGR))

    print(f"✅ Done normalizing for {person_folder}")

print("\n🎉 All 5 persons processed successfully! Normalized images saved in 'normalized_faces/' folder.")
