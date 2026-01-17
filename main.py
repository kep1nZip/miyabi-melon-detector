import cv2
from ultralytics import YOLO
import pygame
import time

# ===============================
# 1. LOAD MODEL YOLO
# ===============================
model = YOLO("yolov8n.pt")

# ===============================
# 2. AUDIO SETUP
# ===============================
pygame.mixer.init()
sound_watermelon = pygame.mixer.Sound("suara1.wav")
sound_other = pygame.mixer.Sound("suara2.wav")

# ===============================
# 3. VIDEO SETUP
# ===============================
video_watermelon = cv2.VideoCapture("video1.mp4")
video_other = cv2.VideoCapture("video2.mp4")

# ===============================
# 4. STATE & TIMER
# ===============================
state = "idle"          # idle | watermelon | other
current_video = None

pending_state = None
state_start_time = 0
DELAY_SECONDS = 1.0     # ⏱️ anti flicker

# ===============================
# 5. KAMERA
# ===============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Kamera tidak terbuka")
    exit()

# ===============================
# 6. LOOP UTAMA
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===============================
    # DETEKSI OBJEK
    # ===============================
    results = model(frame, conf=0.4, verbose=False)

    detected_watermelon = False
    detected_other = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # 🍉 HANYA SEMANGKA
            if label == "watermelon":
                detected_watermelon = True

            # 🍎 BUAH LAIN
            elif label in ["apple", "banana", "orange"]:
                detected_other = True

    # ===============================
    # PRIORITAS STATE
    # ===============================
    if detected_watermelon:
        desired_state = "watermelon"
    elif detected_other:
        desired_state = "other"
    else:
        desired_state = "idle"

    # ===============================
    # DELAY ANTI FLICKER
    # ===============================
    if desired_state != state:
        if pending_state != desired_state:
            pending_state = desired_state
            state_start_time = time.time()
        elif time.time() - state_start_time >= DELAY_SECONDS:
            pygame.mixer.stop()

            if desired_state == "watermelon":
                sound_watermelon.play()
                video_watermelon.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_video = video_watermelon
                print("🍉 WATERMELON")

            elif desired_state == "other":
                sound_other.play()
                video_other.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_video = video_other
                print("🍎 BUAH LAIN")

            else:  # idle
                current_video = None
                print("⬜ IDLE")

            state = desired_state
            pending_state = None
    else:
        pending_state = None

    # ===============================
    # TAMPILKAN VIDEO
    # ===============================
    if current_video is not None:
        ret_vid, vid_frame = current_video.read()
        if not ret_vid:
            current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_vid, vid_frame = current_video.read()

        cv2.imshow("Video Output", vid_frame)
    else:
        # aman tutup window kalau idle
        if cv2.getWindowProperty("Video Output", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Video Output")

    cv2.imshow("Camera", frame)

    # ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ===============================
# 7. CLEAN UP
# ===============================
cap.release()
video_watermelon.release()
video_other.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
