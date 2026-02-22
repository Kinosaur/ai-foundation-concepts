# AI Concepts Term Project - Receipt Object Detection

## Overview

This repository contains the code and resources for a custom Object Detection project using the YOLO architecture. The goal of the model is to detect and extract key fields from receipts. The model was trained to identify four specific classes:
- `quantity_item`
- `product_description`
- `price`
- `total_due_amount`

The repository includes the final exported model, **`best.pt`**, which achieved the best performance metric during training.

## Project Report

The complete rationale, dataset details, methodology, and metrics tracking the progression until we acquired `best.pt` are deeply detailed in the project reports included in this folder:
- `AI Concepts' Term Project Report.pdf`
- `Project_2026.pdf`

Please refer to these documents for a comprehensive theoretical and statistical breakdown of the model's performance.

## Code Files

The `code/` directory contains the scripts and notebooks used throughout the lifecycle of the project. As requested, key files include:

### 1. Training Notebook (`code/Train_yolo26_2026.ipynb`)
This Jupyter Notebook was used to train the YOLO model on the augmented receipt dataset (`YOLODataset_aug`). The notebook outlines the entire pipeline, including:
- Setting up the Ultralytics environment.
- Formatting the dataset YAML for cloud processing.
- Executing the training command for 200 epochs.
- Generating the tracked training output resulting in `best.pt`.

### 2. Inference Script (`code/infer.py`)
This is a lightweight Python script that leverages the compiled `best.pt` model to perform predictions on new data. 
- It loads an image from the `test/` folder (e.g., `test/test2.jpg`).
- It runs the YOLO prediction algorithm.
- It displays and saves the bounding-box annotated results locally.

## Running Inference

You can test the optimized `best.pt` model easily using the provided script. From the root of the project directory, run:

```bash
python code/infer.py
```

*Note: Ensure you have `ultralytics` installed (`pip install ultralytics`) before running the script.*
