import cv2
import time
from ultralytics import YOLO

# ─────────────────────────────────────────
# LOAD MODEL & VIDEO
# ─────────────────────────────────────────
model = YOLO("700images_int8.tflite")

cap = cv2.VideoCapture("video1.mp4")

original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Original video FPS : {original_fps:.2f}")
print(f"Total frames       : {total_frames}")
print(f"─────────────────────────────────────")

# ─────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────
frame_times = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    t_start = time.perf_counter()

    results = model(frame, verbose=False)

    t_end = time.perf_counter()
    inference_ms = (t_end - t_start) * 1000
    frame_times.append(t_end - t_start)

    print(f"Frame {frame_count:04d} | Inference: {inference_ms:.1f} ms")

cap.release()

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
total_time       = sum(frame_times)
avg_fps          = frame_count / total_time
avg_inference    = (total_time / frame_count) * 1000
real_time_factor = avg_fps / original_fps

print(f"─────────────────────────────────────")
print(f"Frames processed   : {frame_count}")
print(f"Total time         : {total_time:.2f} s")
print(f"Original FPS       : {original_fps:.2f}")
print(f"Achieved FPS       : {avg_fps:.2f}")
print(f"Avg inference time : {avg_inference:.1f} ms")
print(f"Real-time factor   : {real_time_factor:.2f}x real time")
print(f"─────────────────────────────────────")
