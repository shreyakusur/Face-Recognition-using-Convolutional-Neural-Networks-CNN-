import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import numpy as np

# Input: normalized images from Part 4
input_folder = "normalized_faces"
output_folder = "augmented_faces"
os.makedirs(output_folder, exist_ok=True)

# Define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,        # rotate image up to 20 degrees
    width_shift_range=0.2,    # shift horizontally by 20%
    height_shift_range=0.2,   # shift vertically by 20%

    brightness_range=[0.7, 1.3],  # change brightness
    horizontal_flip=True,     # flip images horizontally
    fill_mode='nearest'       # fill empty pixels after transformation
)

# Loop through all 5 persons’ folders
for person_folder in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person_folder)
    if not os.path.isdir(person_path):
        continue

    save_path = os.path.join(output_folder, person_folder)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n🔹 Augmenting images for: {person_folder}")

    for img_file in os.listdir(person_path):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_path, img_file)
        img = load_img(img_path)               # load image
        x = img_to_array(img)                  # convert to numpy array
        x = np.expand_dims(x, axis=0)          # add batch dimension

        # Generate 5 augmented versions per image
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=save_path,
                                  save_prefix=person_folder,
                                  save_format='jpg'):
            i += 1
            if i >= 5:
                break  # create only 5 new versions per original image

    print(f"✅ Done augmenting for {person_folder}")

print("\n🎉 All 5 persons processed successfully! Augmented images saved in 'augmented_faces/' folder.")
