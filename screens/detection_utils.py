import time
from collections import defaultdict, deque

from screens.custom_logging import get_logger

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
