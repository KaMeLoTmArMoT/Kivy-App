# Workspace Rules: Kivy App Development Guidelines

This file defines the workspace-scoped rules and instructions for Antigravity (and other agentic assistants) when working on the `Kivy-App` project.

---

## 1. Directory Structure & File Placement

- **View Controllers (Logic)**: Must be placed under `app/screens/view/` with the suffix `_screen.py` (e.g., `analytics_screen.py`).
- **UI Layouts (Declarative)**: Must be placed under `app/ui/` with the suffix `view.kv` or matching naming conventions (e.g., `analyticsview.kv`).
- **Utilities & Helpers**: Must be placed under `app/screens/utils/`.

---

## 2. Kivy Screen Development Rules

### 2.1 View Inheritance
All screens must subclass both `kivy.uix.screenmanager.Screen` and `app.screens.utils.additional.BaseScreen`.
```python
from kivy.uix.screenmanager import Screen
from app.screens.utils.additional import BaseScreen

class MyNewScreen(Screen, BaseScreen):
    pass
```

### 2.2 Navigation and Transition Order
- Do **NOT** set `self.manager.current = ...` directly in the view class code or KV layouts.
- Instead, use the navigation helper methods in `BaseScreen` (e.g. `self.goto_main()`).
- When adding a new screen, always:
  1. Add it to the transition order mapping dictionary `translations` inside `select_direction()` in [additional.py](file:///g:/programming/Kivy-App/app/screens/utils/additional.py).
  2. Implement a corresponding `goto_<name>()` helper method in `BaseScreen`.
  3. Bind navigation buttons in Kivy layouts to calling this method: `on_press: root.parent.parent.goto_<name>()`.

### 2.3 Loading Screen Registration
Every screen must be loaded dynamically by `LoadingScreen` in [loading_screen.py](file:///g:/programming/Kivy-App/app/screens/view/loading_screen.py):
1. Append the load method to `self.modules` in `LoadingScreen.__init__`.
2. Define a `@log_exec_time` decorated loader method in `LoadingScreen` that does `Builder.load_file("app/ui/<name>view.kv")` and `self.manager.add_widget(MyNewScreen(name="<name>view"))`.

### 2.4 Asynchronous Actions & Main Thread Safety
- **UI Freezes**: Never run block-waiting methods, long network requests, or model training/inference synchronously on the main thread.
- **Clock Timers**: Use `Clock.schedule_once` and `Clock.schedule_interval` for small UI updates, asynchronous lists rendering, or lazy evaluations.
- **Threading**: Run background tasks (YOLO exports, PyTorch training/inference, network operations) in separate `threading.Thread` instances.
- **Callbacks**: All callbacks returning from background threads to update UI elements must be wrapped in `Clock.schedule_once`.

---

## 3. Database & Security Isolation

- **No Raw SQL**: Do not execute raw SQL queries inside view code (`app/screens/view/`).
- **DB Helper**: Define a static method helper inside the `DB` class in [db.py](file:///g:/programming/Kivy-App/app/screens/utils/db.py) and invoke it using `self.db.<method_name>()`.
- **Encryption**: Database credentials or sensitive customer entries must be processed using AES EAX encryption. Derived keys must be constructed from the login key using `extend_key(login_key)` in `app.screens.utils.utils.py`.

---

## 4. Coding Style and Preservation

- **Comments**: Retain existing codebase comments and docstrings unless specifically asked to remove or replace them.
- **Type Annotations**: Provide type hints for function signatures and properties where feasible.
- **Linting & Formatting**: Follow pep8 conventions using Ruff. Run `uv run ruff check --fix` and `uv run ruff format`. Ensure there are no unused imports, and avoid long lines.

---

## 5. Verification Requirements

After any code change:
- Run integration tests locally using:
  ```powershell
  uv run pytest --timeout 20 -v -s
  ```
- Do not mark a task completed until all tests pass successfully.
