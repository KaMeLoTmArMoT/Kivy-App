# Integration tests (modules 1–5)

## Prereqs

- Run from repo root (where `pytest.ini` is located).

Recommended flags:

- `-v` = verbose output
- `-s` = show `print()` / console logs
- `--timeout 20` = hard stop if test freezes

Example:

```powershell
pytest --timeout 20 -v -s
```

---

## Run ALL integration tests

Run everything under integration folder:

```powershell
pytest --timeout 20 -v -s .\app\tests\integration\
```

---

## Run module 1–5

```powershell
pytest --timeout 20 -v -s .\app\tests\integration\test_1_screen_navigation.py

pytest --timeout 20 -v -s .\app\tests\integration\test_2_auth_flow.py

pytest --timeout 20 -v -s .\app\tests\integration\test_3_test_main_crud_flow.py

pytest --timeout 20 -v -s .\app\tests\integration\test_4_images_flow.py

pytest --timeout 20 -v -s .\app\tests\integration\test_5_db_images_split.py
```

---

## Run a specific test (node id)

Pytest lets you run a single test by node id using `::` syntax.

### Specific test function

```powershell
pytest --timeout 20 -v -s .\app\tests\integration\test_5_db_images_split.py::test_correct_key_zero_left
```

### Specific test in a class

```powershell
pytest --timeout 20 -v -s .\app\tests\integration\test_5_db_images_split.py::TestDbImagesSplit::test_wrong_key_only_secure_left
```

---

## Debug tips

### Re-run last failed only

```powershell
pytest --timeout 20 -v -s --lf
```

---

## Test Optimization & Speedup Strategies

See [TEST_OPTIMIZATION.md](../../TEST_OPTIMIZATION.md) for techniques to reduce full test suite runtime from ~3 minutes to <30 seconds (mocking PyTorch CPU training, accelerating Kivy Clock intervals, and running `pytest -n auto`).
