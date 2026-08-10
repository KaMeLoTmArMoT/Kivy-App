import os
import time

import cv2
from ultralytics import YOLO


def benchmark_webcam(model_path, camera_index=0, num_frames_to_average=500):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"\n--- Benchmarking {os.path.basename(model_path)} on Webcam ---")
    print("Press 'q' to quit.")

    try:
        model = YOLO(model_path, task="detect")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    total_time = 0

    print("Performing warm-up run...")
    ret, frame = cap.read()
    if ret:
        model(frame, verbose=False)
    print("Warm-up complete. Starting benchmark...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        start_time = time.time()

        results = model(frame, verbose=False)

        end_time = time.time()

        inference_time = end_time - start_time

        if frame_count < num_frames_to_average:
            total_time += inference_time
            frame_count += 1

        annotated_frame = results[0].plot()

        fps = 1 / inference_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow(f"Webcam Benchmark - {os.path.basename(model_path)}", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if frame_count > 0:
        average_fps = frame_count / total_time
        print(f"\n--- Results for {os.path.basename(model_path)} ---")
        print(f"Average FPS over {frame_count} frames: {average_fps:.2f}")
        print("---------------------------------------\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    BASE_PATH = "/projects_detection/default"
    # MODEL_NAME = "yolo11n"
    MODEL_NAME = "yolo11x"

    # model_to_test = os.path.join(BASE_PATH, f"{MODEL_NAME}.engine")
    # model_to_test = os.path.join(BASE_PATH, f"{MODEL_NAME}.onnx")
    model_to_test = os.path.join(BASE_PATH, f"{MODEL_NAME}.pt")

    benchmark_webcam(model_to_test)
