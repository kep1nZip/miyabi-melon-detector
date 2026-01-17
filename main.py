import cv2
from ultralytics import YOLO
import pygame
import time

# ===============================
# 1. LOAD MODEL YOLO
# ===============================
model = YOLO("yolov8n.pt")  # auto download jika belum ada

# ===============================
# 2. AUDIO SETUP
# ===============================
pygame.mixer.init()

sound_melon = pygame.mixer.Sound("suara1.wav")
sound_other = pygame.mixer.Sound("suara2.wav")

# ===============================
# 3. VIDEO SETUP
# ===============================
video_melon = cv2.VideoCapture("video1.mp4")
video_other = cv2.VideoCapture("video2.mp4")

current_video = video_other
last_state = None

# ===============================
# 4. KAMERA
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera tidak terbuka")
    exit()
    
# ===============================
# 5. LOOP UTAMA
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek
    results = model(frame, conf=0.4)

    detected_melon = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "watermelon":
                detected_melon = True

    # ===============================
    # 6. LOGIKA PERUBAHAN KONDISI
    # ===============================
    if detected_melon and last_state != "melon":
        pygame.mixer.stop()
        sound_melon.play()
        video_melon.set(cv2.CAP_PROP_POS_FRAMES, 0)
        current_video = video_melon
        last_state = "melon"
        print("🍉 Melon terdeteksi")

    elif not detected_melon and last_state != "other":
        pygame.mixer.stop()
        sound_other.play()
        video_other.set(cv2.CAP_PROP_POS_FRAMES, 0)
        current_video = video_other
        last_state = "other"
        print("❌ Bukan melon")

    # ===============================
    # 7. TAMPILKAN VIDEO
    # ===============================
    ret_vid, vid_frame = current_video.read()
    if not ret_vid:
        current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_vid, vid_frame = current_video.read()

    cv2.imshow("Video Output", vid_frame)
    cv2.imshow("Camera", frame)

    # ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ===============================
# 8. CLEAN UP
# ===============================
cap.release()
video_melon.release()
video_other.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
