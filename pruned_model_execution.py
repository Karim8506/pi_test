import cv2
import time
from ultralytics import YOLO

# LOAD MODEL & VIDEO
model = YOLO("700images_pruned.pt")
cap = cv2.VideoCapture("video1.mp4")

original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Original video FPS : {original_fps:.2f}")
print(f"Total frames       : {total_frames}")
print("─────────────────────────────────────")

frame_times = []
frame_count = 0

t_total_start = time.perf_counter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    t_start = time.perf_counter()
    results = model(frame, verbose=False, device="cpu")
    t_end = time.perf_counter()

    inference_ms = (t_end - t_start) * 1000
    frame_times.append(t_end - t_start)

    print(f"Frame {frame_count:04d} | Inference: {inference_ms:.1f} ms")

cap.release()

t_total_end = time.perf_counter()

# SUMMARY
inference_total_time = sum(frame_times)
wall_time = t_total_end - t_total_start

avg_inference_ms = (inference_total_time / frame_count) * 1000
inference_fps = frame_count / inference_total_time
pipeline_fps = frame_count / wall_time
real_time_factor = pipeline_fps / original_fps

print("─────────────────────────────────────")
print(f"Frames processed    : {frame_count}")
print(f"Inference-only time : {inference_total_time:.2f} s")
print(f"Wall time           : {wall_time:.2f} s")
print(f"Original FPS        : {original_fps:.2f}")
print(f"Inference FPS       : {inference_fps:.2f}")
print(f"Pipeline FPS        : {pipeline_fps:.2f}")
print(f"Avg inference time  : {avg_inference_ms:.1f} ms")
print(f"Real-time factor    : {real_time_factor:.2f}x real time")
print("─────────────────────────────────────")
