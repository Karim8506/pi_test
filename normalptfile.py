import time
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────
# LOAD MODEL & VIDEO
# ─────────────────────────────────────────
model = YOLO("700images.pt")

# ─────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────
results = model("video1.mp4", verbose=False, stream=True)

frame_times = []
frame_count = 0

t_total_start = time.perf_counter()

for result in results:
    frame_count += 1
    t_start = time.perf_counter()
    _ = result.boxes
    t_end = time.perf_counter()
    inference_ms = (t_end - t_start) * 1000
    frame_times.append(t_end - t_start)
    print(f"Frame {frame_count:04d} | Inference: {inference_ms:.1f} ms")

t_total_end = time.perf_counter()

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
original_fps     = 30.0  # set your video FPS here
total_time       = sum(frame_times)
avg_fps          = frame_count / (t_total_end - t_total_start)
avg_inference    = (total_time / frame_count) * 1000
real_time_factor = avg_fps / original_fps

print(f"─────────────────────────────────────")
print(f"Frames processed   : {frame_count}")
print(f"Total time         : {(t_total_end - t_total_start):.2f} s")
print(f"Original FPS       : {original_fps:.2f}")
print(f"Achieved FPS       : {avg_fps:.2f}")
print(f"Avg inference time : {avg_inference:.1f} ms")
print(f"Real-time factor   : {real_time_factor:.2f}x real time")
print(f"─────────────────────────────────────")
