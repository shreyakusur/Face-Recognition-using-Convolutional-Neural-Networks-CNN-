import cv2
import os

# Input and output folders
input_folder = "cropped_faces"      # from Part 2
output_folder = "resized_faces"     # resized images saved here
os.makedirs(output_folder, exist_ok=True)

# Choose target resize dimension (adjust as needed)
resize_dim = (128, 128)  # options: (128,128), (64,64), (32,32)

# Loop through all 5 persons' folders
for person_folder in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person_folder)
    if not os.path.isdir(person_path):
        continue

    save_path = os.path.join(output_folder, person_folder)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n🔹 Resizing images for: {person_folder}")

    for img_file in os.listdir(person_path):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)

        # Resize to desired dimensions
        resized_img = cv2.resize(img, resize_dim)

        # Save the resized image
        cv2.imwrite(os.path.join(save_path, img_file), resized_img)

    print(f"✅ Done resizing for {person_folder}")

print("\n🎉 All 5 persons processed successfully! Resized images saved in 'resized_faces/' folder.")
