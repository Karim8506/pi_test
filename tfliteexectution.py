import cv2
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "700images_int8.tflite"
VIDEO_PATH = "video1.mp4"

def main():
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    frame_count = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # minimal resize to model input
        h = input_details[0]["shape"][1]
        w = input_details[0]["shape"][2]
        img = cv2.resize(frame, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if input_details[0]["dtype"] == np.float32:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(input_details[0]["dtype"])

        img = np.expand_dims(img, axis=0)

        t_start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        t_end = time.perf_counter()

        print(f"Frame {frame_count:04d} | Inference: {(t_end - t_start)*1000:.1f} ms | Output shape: {output.shape}")

    cap.release()

    total = time.perf_counter() - t0
    print("─────────────────────────────────────")
    print(f"Frames processed : {frame_count}")
    print(f"Wall time        : {total:.2f} s")
    print(f"Pipeline FPS     : {frame_count / total:.2f}" if total > 0 else "Pipeline FPS     : 0.00")
    print("─────────────────────────────────────")

if __name__ == "__main__":
    main()
