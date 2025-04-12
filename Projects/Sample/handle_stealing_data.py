import cv2
import os

video_data_path = '0621video\\video_sample_data\\video\\C041_A30_SY32_P07_S06_02DAS.mp4'

cap = cv2.VideoCapture(video_data_path)

img_count = 0

folder_name = video_data_path.split("\\")[-1]
folder_name = folder_name.replace(".mp4", "")
os.makedirs(f"./AI_hub_frame_dataset/{folder_name}", exist_ok=True)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if img_count % 15 == 0:
        img_filename = f"./AI_hub_frame_dataset/{folder_name}/frame_{img_count:04}.png"
        cv2.imwrite(img_filename, frame)

    img_count += 1

cap.release()