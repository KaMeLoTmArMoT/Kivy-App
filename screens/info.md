# Detection projects structure:
```
.../app.py
├── projects_detection
│   ├── {project name}
│   │   ├── dataset
│   │   │   ├── raw
│   │   │   │   ├── annotations
│   │   │   │   │   ├── classes.txt
│   │   │   │   │   ├── {annotation}.txt
│   │   │   │   │   └── ...
│   │   │   │   ├── images
│   │   │   │   │   ├── {img}.jpg (TODO: check .png support)
│   │   │   │   │   └── ...
│   │   │   │   └── out <- temporary save train test split
│   │   │   │       ├── test
│   │   │   │       │   ├── {img1}.jpg
│   │   │   │       │   ├── {img1}.txt
│   │   │   │       │   └── ...
│   │   │   │       └── train
│   │   │   │           ├── {img2}.jpg
│   │   │   │           ├── {img2}.txt
│   │   │   │           └── ...
│   │   │   │
│   │   │   ├── train
│   │   │   │   ├── images
│   │   │   │   │   ├── {img1}.jpg
│   │   │   │   │   └── ...
│   │   │   │   ├── labels
│   │   │   │   │   ├── {img1}.txt
│   │   │   │   │   └── ...
│   │   │   │   └── labels.cache
│   │   │   │
│   │   │   ├── val
│   │   │   │   ├── images
│   │   │   │   │   ├── {img2}.jpg
│   │   │   │   │   └── ...
│   │   │   │   ├── labels
│   │   │   │   │   ├── {img2}.txt
│   │   │   │   │   └── ...
│   │   │   │   │
│   │   │   │   └── labels.cache
│   │   │   │
│   │   │   └── custom_dataset.yaml
│   │   │
│   │   └── yolov8{n/s/m/l/x}.pt  <- trained model
│   │
│   ├── {project 2 name}
...
```