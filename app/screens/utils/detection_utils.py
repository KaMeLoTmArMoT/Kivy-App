import time
from collections import defaultdict, deque

from app.screens.utils.custom_logging import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    def __init__(self, max_len=100):
        self.timings = defaultdict(lambda: deque(maxlen=max_len))

    def add_time(self, key: str, duration_ms: float):
        self.timings[key].append(duration_ms)

    @staticmethod
    def elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000

    def record(self, label: str, start_time: float):
        self.add_time(label, self.elapsed_ms(start_time))

    def report(self):
        report_lines = []

        global_avg = (
            sum(self.timings["global"]) / len(self.timings["global"])
            if "global" in self.timings and self.timings["global"]
            else 0.0
        )
        report_lines.append(f"{'GLOBAL':<10} | Avg: {global_avg:7.1f} ms | 100.0%")

        measured_keys = [k for k in self.timings if k != "global"]
        total_measured_time = 0.0
        per_module_lines = []

        for key in measured_keys:
            values = self.timings[key]
            total = sum(values)
            avg = total / len(values) if values else 0.0
            total_measured_time += avg
            percent = (avg / global_avg * 100.0) if global_avg else 0.0
            per_module_lines.append(
                f"{key.upper():<10} | Avg: {avg:7.1f} ms | {percent:5.1f}%"
            )

        other_time = global_avg - total_measured_time
        if other_time < 0:
            other_time = 0.0
        other_percent = (other_time / global_avg * 100.0) if global_avg else 0.0
        per_module_lines.append(
            f"{'OTHER':<10} | Avg: {other_time:7.1f} ms | {other_percent:5.1f}%"
        )

        if global_avg > 0:
            avg_fps = 1000.0 / global_avg
            per_module_lines.append(f"{'FPS':<10} | Est: {avg_fps:7.1f}")

        report_lines.extend(per_module_lines)
        return "\n".join(report_lines)

    def clear_timings(self):
        logger.warning("Clearing timings")
        for key in self.timings:
            self.timings[key].clear()


def split_detection_dataset(projects_folder: str, active_project: str):
    import os
    import shutil

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.error(
            "scikit-learn not installed. Install scikit-learn to use dataset split."
        )
        return

    pth_annotations = os.path.join(
        projects_folder, active_project, "dataset", "raw", "annotations"
    )
    pth_images = os.path.join(
        projects_folder, active_project, "dataset", "raw", "images"
    )
    logger.info(f"split: {pth_annotations=}, {pth_images=}")

    if not os.path.exists(pth_annotations) or not os.path.exists(pth_images):
        logger.error("Annotations or images path does not exist.")
        return

    annotations = os.listdir(pth_annotations)
    if "classes.txt" in annotations:
        annotations.remove("classes.txt")
    images = os.listdir(pth_images)
    logger.info(f"all: {len(annotations)=}, {len(images)=}")

    selected_images = []
    for annotation in annotations:
        name = annotation.replace(".txt", ".png")
        if name in images:
            selected_images.append(name)

    logger.info(f"clear: {len(annotations)=}, {len(selected_images)=}")

    if len(selected_images) == 0:
        logger.warning("No matching images and annotations found.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        selected_images, annotations, test_size=0.2
    )
    logger.info(f"{len(X_train)=} {len(y_train)=}\n{len(X_test)=} {len(y_test)=}")

    out_train = os.path.join(
        projects_folder, active_project, "dataset", "raw", "out", "train"
    )
    out_test = os.path.join(
        projects_folder, active_project, "dataset", "raw", "out", "val"
    )

    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)
    os.makedirs(os.path.join(out_train, "labels"), exist_ok=True)
    os.makedirs(os.path.join(out_train, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_test, "labels"), exist_ok=True)
    os.makedirs(os.path.join(out_test, "images"), exist_ok=True)

    for img, ann in zip(X_train, y_train):
        shutil.copy(
            os.path.join(pth_annotations, ann),
            os.path.join(out_train, "labels", ann),
        )
        shutil.copy(
            os.path.join(pth_images, img), os.path.join(out_train, "images", img)
        )

    for img, ann in zip(X_test, y_test):
        shutil.copy(
            os.path.join(pth_annotations, ann),
            os.path.join(out_test, "labels", ann),
        )
        shutil.copy(
            os.path.join(pth_images, img), os.path.join(out_test, "images", img)
        )

    class_file = os.path.join(pth_annotations, "classes.txt")
    logger.debug(f"{class_file}")
    if os.path.exists(class_file):
        with open(class_file, "r") as f:
            classes = f.read().split("\n")
            if "" in classes:
                classes.remove("")
            logger.debug(f"{classes=}, {len(classes)=}")
    else:
        classes = []

    yaml_file = os.path.join(
        projects_folder, active_project, "dataset", "custom_dataset.yaml"
    )
    with open(yaml_file, "w") as f:
        f.write("train: ./train\n")
        f.write("val: ./val\n")
        f.write("\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("\n")
        f.write(f"names: {classes}")

    shutil.move(
        out_train,
        os.path.join(projects_folder, active_project, "dataset"),
    )
    shutil.move(out_test, os.path.join(projects_folder, active_project, "dataset"))
