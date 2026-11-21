import os
import time

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def export(model_name="yolo11n"):
    base_pth = "/home/lv-user187/PycharmProjects/Kivy-App/projects_detection/default"
    export_model_path = os.path.join(base_pth, f"{model_name}.pt")

    model = YOLO(export_model_path)
    model.export(format="tensorrt", quantize=16, simplify=True)
    model.export(format="openvino", quantize=16)
    print("Model exported successfully to TensorRT/TARGET format!")


def benchmark(model_path, dummy_input, log_file, num_runs=500):
    if not model_path or not os.path.exists(model_path):
        print(f"Skipping benchmark for non-existent model: {model_path}")
        return

    print(f"\n--- Benchmarking {os.path.basename(model_path)} ---")

    model = YOLO(model_path, task="detect")
    try:
        model.fuse()
        print("Fuse ok")
    except Exception as e:
        print(f"Failed to fuse model {model_path}\n{e}")

    print("Performing warm-up run...")
    model(dummy_input, verbose=False)
    print("Warm-up complete.")

    start_time = time.time()
    for _ in tqdm(range(num_runs)):
        model(dummy_input, verbose=False)
        # model(dummy_input, verbose=False, device="intel:cpu")
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


def run_test(model_name="yolo11n", num_runs=500):
    log_file = "../app/screens/speed_benchmark.log"

    base_pth = "/home/lv-user187/PycharmProjects/Kivy-App/projects_detection/default"

    pt_model_path = os.path.join(base_pth, f"{model_name}.pt")
    onnx_model_path = os.path.join(base_pth, f"{model_name}.onnx")
    tensorrt_model_path = os.path.join(base_pth, f"{model_name}.engine")
    vino_model_path = os.path.join(base_pth, f"{model_name}_openvino_model")

    input_image = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)

    with open(log_file, "a") as f:
        f.write("\n\n\n")

    model_paths = [pt_model_path, onnx_model_path, tensorrt_model_path, vino_model_path]
    for model_path in model_paths:
        try:
            benchmark(model_path, input_image, log_file, num_runs)
            time.sleep(5)
        except Exception as e:
            print(f"Skipping benchmark for model: {model_path}\nwith error: {e}")

    print(f"\nBenchmark complete. Results saved to {log_file}")


if __name__ == "__main__":
    model_mame = "yolo11n"

    # export(model_mame)
    run_test(model_mame, num_runs=200)
