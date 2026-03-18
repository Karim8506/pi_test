import cv2
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "700images_int8.tflite"
VIDEO_PATH = "video1.mp4"

CONF_THRES = 0.25
IOU_THRES = 0.45

# Put your class names here if you know them
CLASS_NAMES = ["object"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []

    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        rest = []

        for i in idxs[1:]:
            if iou(boxes[current], boxes[i]) < iou_thres:
                rest.append(i)

        idxs = np.array(rest, dtype=np.int64)

    return keep


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_w, new_h = new_shape

    r = min(new_w / w, new_h / h)
    resized_w = int(round(w * r))
    resized_h = int(round(h * r))

    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return out, r, left, top


def preprocess(frame, input_h, input_w, input_dtype):
    img, scale, pad_x, pad_y = letterbox(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if input_dtype == np.float32:
        tensor = img.astype(np.float32) / 255.0
    else:
        tensor = img.astype(input_dtype)

    tensor = np.expand_dims(tensor, axis=0)
    return tensor, scale, pad_x, pad_y


def decode_yolo_output(output, orig_w, orig_h, scale, pad_x, pad_y):
    """
    Handles common YOLO TFLite detect outputs:
    - (1, N, 4 + nc)
    - (1, 4 + nc, N)
    """
    pred = np.squeeze(output)

    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected output shape after squeeze: {pred.shape}")

    # Convert to shape (N, 4 + nc)
    if pred.shape[0] < pred.shape[1]:
        # likely (channels, N) -> transpose
        pred = pred.T

    if pred.shape[1] < 6:
        raise RuntimeError(f"Output does not look like YOLO detect output: {pred.shape}")

    boxes = []
    scores = []
    class_ids = []

    for row in pred:
        cx, cy, w, h = row[:4]
        class_scores = row[4:]

        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])

        if score < CONF_THRES:
            continue

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Undo letterbox
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)

    keep = nms(boxes, scores, IOU_THRES)

    final = []
    for i in keep:
        final.append((boxes[i], scores[i], class_ids[i]))

    return final


def draw_detections(frame, detections):
    for box, score, class_id in detections:
        x1, y1, x2, y2 = map(int, box)
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        label = f"{name} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main():
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    _, input_h, input_w, _ = input_shape

    print("Model input shape :", input_shape)
    print("Model input dtype :", input_dtype)
    print("Output tensors    :", [o["shape"] for o in output_details])

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original video FPS : {original_fps:.2f}")
    print(f"Total frames       : {total_frames}")
    print("─────────────────────────────────────")

    frame_count = 0
    frame_times = []
    wall_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        orig_h, orig_w = frame.shape[:2]

        input_tensor, scale, pad_x, pad_y = preprocess(
            frame, input_h, input_w, input_dtype
        )

        t0 = time.perf_counter()

        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        detections = decode_yolo_output(output, orig_w, orig_h, scale, pad_x, pad_y)

        t1 = time.perf_counter()
        dt = t1 - t0
        frame_times.append(dt)

        draw_detections(frame, detections)

        print(
            f"Frame {frame_count:04d} | Inference: {dt * 1000:.1f} ms | Detections: {len(detections)}"
        )

        cv2.imshow("TFLite YOLO", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_count == 0:
        print("No frames processed.")
        return

    wall_total = time.perf_counter() - wall_start
    inference_total = sum(frame_times)

    avg_inference_ms = (inference_total / frame_count) * 1000
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
    print(f"Avg inference time : {avg_inference_ms:.1f} ms")
    print(f"Real-time factor   : {real_time_factor:.2f}x real time")
    print("─────────────────────────────────────")


if __name__ == "__main__":
    main()
