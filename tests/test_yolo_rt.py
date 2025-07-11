import os
import time

import numpy as np
from ultralytics import YOLO

def export(model_name="yolo11n"):
    base_pth = "/home/lv-user187/PycharmProjects/Kivy-App/projects_detection/default"
    export_model_path = os.path.join(base_pth, f"{model_name}.pt")

    model = YOLO(export_model_path)
    # model.export(format='tensorrt')
    model.export(format='tensorrt', half=True, simplify=True)
    print("Model exported successfully to TensorRT format!")


def benchmark(model_path, dummy_input, log_file, num_runs=500):
    if not model_path or not os.path.exists(model_path):
        print(f"Skipping benchmark for non-existent model: {model_path}")
        return

    print(f"\n--- Benchmarking {os.path.basename(model_path)} ---")

    model = YOLO(model_path, task="detect")

    print("Performing warm-up run...")
    model(dummy_input, verbose=False)
    print("Warm-up complete.")

    start_time = time.time()
    for _ in range(num_runs):
        model(dummy_input, verbose=False)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_ms = (total_time / num_runs) * 1000
    fps = 1 / (total_time / num_runs)

    result_str = (
        f"Model: {os.path.basename(model_path)}\n"
        f"Average inference time: {avg_time_ms:.2f} ms\n"
        f"Frames Per Second (FPS): {fps:.2f}\n"
        f"---------------------------------------\n"
    )

    print(result_str)
    with open(log_file, "a") as f:
        f.write(result_str)


def run_test(model_name="yolo11n"):
    log_file = "../screens/speed_benchmark.log"

    base_pth = "/home/lv-user187/PycharmProjects/Kivy-App/projects_detection/default"

    pt_model_path = os.path.join(base_pth, f"{model_name}.pt")
    onnx_model_path = os.path.join(base_pth, f"{model_name}.onnx")
    tensorrt_model_path = os.path.join(base_pth, f"{model_name}.engine")

    input_image = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)

    with open(log_file, "a") as f:
        f.write("\n\n\n")

    benchmark(pt_model_path, input_image, log_file)
    time.sleep(5)
    benchmark(onnx_model_path, input_image, log_file)
    time.sleep(5)
    benchmark(tensorrt_model_path, input_image, log_file)

    print(f"\nBenchmark complete. Results saved to {log_file}")


if __name__ == "__main__":
    # export("yolo11x")
    run_test("yolo11x")
