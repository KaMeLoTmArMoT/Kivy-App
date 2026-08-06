# How to Speed Up Kivy-App Tests

**Current Execution Time**: ~181 seconds (3 minutes) for 72 tests (74.6% code coverage).

---

## Key Bottleneck Areas & Solutions

### 1. Mock PyTorch Model Training in Integration Tests (Estimated Savings: ~120s)
- **Problem**: 	est_6_ml_project_model_flow.py executes real PyTorch MobileNetV3 training loops on the CPU during full integration test flows.
- **Fix**:
  - In integration/UI flow tests, mock 	rain() / it() methods to return dummy loss/metrics instantly.
  - Or configure epochs=1 and pass synthetic pre-computed model weights in APP_ENV=test mode.
  - Keep real PyTorch training restricted to isolated unit tests.

### 2. Accelerate Kivy Clock Delay Intervals (wait_until) (Estimated Savings: ~30s)
- **Problem**: Helper functions like wait_until(predicate, timeout=20) poll Kivy frames using real wall-clock time (	ime.monotonic()), resulting in idle wait frames.
- **Fix**:
  - In headless test execution (APP_ENV=test), increase the frame step multiplier in drain(step_frames) or mock Clock.tick() to trigger handlers without waiting for real-time clock delays.

### 3. Parallel Test Execution (pytest-xdist) (Estimated Speedup: 3x)
- **Problem**: Tests execute sequentially in a single process.
- **Fix**:
  - Install pytest-xdist dev dependency: uv add --dev pytest-xdist
  - Run pytest across CPU cores in parallel:
    `powershell
    uv run pytest -n auto
    `

---

## Recommended Daily Command

For fast local developer feedback during UI feature development:
`powershell
uv run pytest app/tests/integration/test_1_screen_navigation.py test_2_auth_flow.py
`
