import cv2
import datetime

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Codec & initialize
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # use uppercase MP4V
out = None
recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Display timestamp on the frame
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # If recording, write frame
    if recording:
        out.write(frame)
        cv2.putText(frame, "● REC", (1100, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show window
    cv2.imshow("Webcam Recorder", frame)

    # Keyboard control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('r'):
        if not recording:
            print("Recording started...")
            recording = True
            # Create new file with timestamp name
            filename = datetime.datetime.now().strftime("Recording_%Y%m%d_%H%M%S.mp4")
            out = cv2.VideoWriter(filename, fourcc, 30.0,
                                  (int(cap.get(3)), int(cap.get(4))))
        else:
            print("Already recording...")
    elif key == ord('s'):
        if recording:
            print("Recording stopped.")
            recording = False
            out.release()
        else:
            print("Not recording currently.")
# Release all
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()