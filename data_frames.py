import cv2
import os

# Paths
dataset_path = "dataset"             # folder containing input videos
output_path = "frames"               # folder to save extracted frames
os.makedirs(output_path, exist_ok=True)

frames_per_video = 120  # number of frames to extract per video

# Loop through all videos
for video_file in os.listdir(dataset_path):
    if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue  # skip non-video files

    video_name = os.path.splitext(video_file)[0]  # get person name (e.g., "person1")
    person_folder = os.path.join(output_path, video_name)
    os.makedirs(person_folder, exist_ok=True)

    # Open the video
    video_path = os.path.join(dataset_path, video_file)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // frames_per_video)  # spacing between frames

    count, saved = 0, 0
    while cap.isOpened() and saved < frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at intervals
        if count % frame_interval == 0:
            img_name = f"{video_name} - img-{saved+1}.jpg"
            img_path = os.path.join(person_folder, img_name)
            cv2.imwrite(img_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"✅ Extracted {saved} frames for {video_name}")

print("\nAll videos processed successfully!")
