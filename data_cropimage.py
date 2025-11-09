import cv2
import os

# Input: grayscale images from Part 1
input_folder = "gray_images"        # each subfolder = one person
output_folder = "cropped_faces"     # folder to save cropped faces
os.makedirs(output_folder, exist_ok=True)

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Loop through all 5 persons' folders
for person_folder in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person_folder)
    if not os.path.isdir(person_path):
        continue  # skip non-folder files

    # Create output folder for this person
    save_path = os.path.join(output_folder, person_folder)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n🔹 Processing: {person_folder}")

    # Loop through all images of this person
    for img_file in os.listdir(person_path):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        # If a face is detected, crop and save
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_crop = img[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(save_path, img_file), face_crop)
                break  # only save the first detected face
        else:
            print(f"⚠️ No face detected in {img_file}")

    print(f"✅ Done: {person_folder}")

print("\n🎉 All 5 persons processed successfully! Cropped face images saved in 'cropped_faces/' folder.")
