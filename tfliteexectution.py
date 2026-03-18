import cv2
import time
from ultralytics import YOLO

# LOAD MODEL & VIDEO
model = YOLO("700images_int8.tflite")

cap = cv2.VideoCapture("video1.mp4")
if not cap.isOpened():
    raise RuntimeError("Could not open video1.mp4")

original_fps = cap.get(cv2.CAP_PROP_FPS)
if original_fps <= 0:
    original_fps = 30.0  # fallback if metadata is missing

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Original video FPS : {original_fps:.2f}")
print(f"Total frames       : {total_frames}")
print("─────────────────────────────────────")

# RUN INFERENCE
frame_times = []
frame_count = 0

wall_start = time.perf_counter()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        t_start = time.perf_counter()
        results = model(frame, verbose=False)
        t_end = time.perf_counter()

        inference_s = t_end - t_start
        inference_ms = inference_s * 1000
        frame_times.append(inference_s)

        print(f"Frame {frame_count:04d} | Inference: {inference_ms:.1f} ms")

finally:
    cap.release()
    cv2.destroyAllWindows()

# SUMMARY
wall_end = time.perf_counter()

if frame_count == 0:
    print("No frames were processed.")
else:
    inference_total = sum(frame_times)
    wall_total = wall_end - wall_start

    avg_inference = (inference_total / frame_count) * 1000
    inference_fps = frame_count / inference_total if inference_total > 0 else 0.0
    pipeline_fps = frame_count / wall_total if wall_total > 0 else 0.0
    real_time_factor = pipeline_fps / original_fps if original_fps > 0 else 0.0

    print("─────────────────────────────────────")
    print(f"Frames processed   : {frame_count}")
    print(f"Inference time     : {inference_total:.2f} s")
    print(f"Wall time          : {wall_total:.2f} s")
    print(f"Original FPS       : {original_fps:.2f}")
    print(f"Inference FPS      : {inference_fps:.2f}")
    print(f"Pipeline FPS       : {pipeline_fps:.2f}")
    print(f"Avg inference time : {avg_inference:.1f} ms")
    print(f"Real-time factor   : {real_time_factor:.2f}x real time")
    print("─────────────────────────────────────")
