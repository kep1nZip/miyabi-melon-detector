import cv2
from ultralytics import YOLO
import pygame
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

model = YOLO("runs/detect/train2/weights/best.pt")

pygame.mixer.init()
sound_melon = pygame.mixer.Sound(os.path.join(ASSETS_DIR, "suara1.wav"))
sound_other = pygame.mixer.Sound(os.path.join(ASSETS_DIR, "suara2.wav"))
video_melon = cv2.VideoCapture(os.path.join(ASSETS_DIR, "video1.mp4"))
video_other = cv2.VideoCapture(os.path.join(ASSETS_DIR, "video2.mp4"))

state = "idle"
current_video = None

pending_state = None
state_start_time = 0
DELAY_SECONDS = 1.0 

melon_frame_count = 0
FRAME_CONFIRM = 2
NO_OBJECT_COUNT = 0
NO_OBJECT_THRESHOLD = 5

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Kamera tidak terbuka")
    exit()

print("🟢 START → IDLE")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3, iou=0.5, verbose=False)

    detected_melon = False
    detected_anything = False

    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            detected_anything = True
            detected_melon = True
            break

    if detected_melon:
        melon_frame_count += 1
        NO_OBJECT_COUNT = 0
    else:
        melon_frame_count = 0
        NO_OBJECT_COUNT += 1

    if melon_frame_count >= FRAME_CONFIRM:
        desired_state = "melon"
    elif NO_OBJECT_COUNT >= NO_OBJECT_THRESHOLD:
        desired_state = "idle"
    else:
        desired_state = state

    if desired_state != state:
        if pending_state != desired_state:
            pending_state = desired_state
            state_start_time = time.time()
        elif time.time() - state_start_time >= DELAY_SECONDS:
            pygame.mixer.stop()

            if desired_state == "melon":
                sound_melon.play()
                video_melon.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_video = video_melon
                print("🍉 MELON")

            elif desired_state == "idle":
                current_video = None
                print("⬜ IDLE")

            state = desired_state
            pending_state = None
    else:
        pending_state = None

    if current_video is not None:
        ret_vid, vid_frame = current_video.read()
        if not ret_vid:
            current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_vid, vid_frame = current_video.read()

        cv2.imshow("Video Output", vid_frame)
    else:
        if cv2.getWindowProperty("Video Output", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Video Output")

    cv2.imshow("Camera", frame)

    # ESC keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

#CLEAN UP

cap.release()
video_melon.release()
video_other.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
