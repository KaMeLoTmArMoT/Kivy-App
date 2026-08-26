# Test Suite & Performance

## Overview

The test suite is split into fast and slow gates to provide immediate feedback on core functionality while keeping heavy ML workloads isolated:

- **Fast Gate (`not slow`)**: Unit tests and integration flows (navigation, authentication, CRUD, image browsing, encrypted DB). Runtime: ~15-20s.
- **Slow Gate (`slow`)**: ML workflows (PyTorch/torchvision model creation, training loops, evaluation, TensorBoard output).

## Running Tests

### Fast Gate (Standard CI / Local Dev)
```powershell
uv run pytest --timeout 30 -m "not slow" -q
```

### ML Gate
```powershell
uv run pytest --timeout 30 -m slow -q
```

### Individual Modules / Tests
```powershell
# Run a specific module
uv run pytest --timeout 20 -v -s .\app\tests\integration\test_1_screen_navigation.py

# Run a specific test function
uv run pytest --timeout 20 -v -s .\app\tests\integration\test_5_db_images_split.py::test_correct_key_zero_left

# Re-run only failed tests
uv run pytest --timeout 20 -v -s --lf
```

## Performance & Execution Rules

- **`APP_ENV=test`**: Skips full PyTorch training loops in `classification.py` while verifying TensorBoard logging and model pipeline health.
- **Clock & Image Bounds**: Image loading is incremental and bound to Kivy `Clock` events to prevent UI blocking.
- **Batched DB Setup**: Database schema initialization uses batched default inserts.
- **Serial Execution**: Test database (`app.db`) and ML workspaces are shared during test runs. Run tests serially; enable parallel execution (`-n auto`) only if workers are isolated with independent workspaces/databases.
