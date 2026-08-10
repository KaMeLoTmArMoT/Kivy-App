# Test Performance

The fast integration gate covers navigation, authentication, CRUD, image workflows, and encrypted DB images. The ML flow is separate because model creation, prediction, TensorBoard output, and evaluation load PyTorch and torchvision.

## Current Behavior

- `APP_ENV=test` skips the full PyTorch training loop in `classification.py` while still writing a TensorBoard scalar.
- Image loading is incremental and bounded by Kivy `Clock` events.
- Database schema/default setup uses batched default configuration inserts.

## Recommended Commands

Fast feedback:

```powershell
uv run pytest --timeout 30 -m "not slow" -q
```

ML flow:

```powershell
uv run pytest --timeout 30 -m slow -q
```

The slow test should remain isolated from daily UI feedback. Parallel execution should only be enabled after each worker uses an isolated DB and project workspace.
