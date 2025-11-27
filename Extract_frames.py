import cv2
import os

def extract_frames(video_path, output_folder, interval_sec=2):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps * interval_sec)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"\n Done! Extracted {saved_count} frames from the video.")

if __name__ == "__main__":
    video_path = "Data/Video/input.mp4"   
    output_folder = "Data/Frames"
    extract_frames(video_path, output_folder)
