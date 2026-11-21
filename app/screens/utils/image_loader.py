from collections import deque
from collections.abc import Callable

from kivy.clock import Clock


class ImageLoadController:
    """Render image paths incrementally without blocking the Kivy event loop."""

    def __init__(
        self,
        grid,
        progress_bar,
        create_widget: Callable[[str], object],
        on_finish: Callable[[], None],
        should_stop: Callable[[], bool],
    ):
        self.grid = grid
        self.progress_bar = progress_bar
        self.create_widget = create_widget
        self.on_finish = on_finish
        self.should_stop = should_stop
        self.pending = deque()
        self.event = None

    def start(self, paths: list[str]):
        self.stop()
        self.pending = deque(paths)
        self.progress_bar.value = 0
        self.progress_bar.max = len(self.pending)
        if self.pending:
            self.event = Clock.schedule_interval(self.step, 0.001)
        return self.event

    def stop(self) -> None:
        if self.event is not None:
            self.event.cancel()
            self.event = None

    def step(self, *_):
        if not self.pending or self.should_stop():
            self.stop()
            self.on_finish()
            return

        self.progress_bar.value += 1
        self.grid.add_widget(self.create_widget(self.pending.pop()))
