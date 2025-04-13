# Classroom Object Detector

A fun and simple machine learning project designed for kids to learn how computers can recognize objects like pens, glasses, erasers, and notebooks using a webcam!

## Project Overview

In this project, kids will:
- Collect images of objects using a webcam
- Train a machine learning model to recognize them
- Test the model in real-time during a live demo for parents!

---


---

## What the kids will code

| File             | Description |
|------------------|-------------|
| `save_images.py` | Code to take pictures of classroom objects using OpenCV and organize them into folders
| `train_model.py` | Code to train a simple machine learning model (like a neural network) using a library like TensorFlow. Starter code is provided with comments and blanks to fill in.
| `preprocess.py` | processes the images collected for training. It resizes the images, flattens them into 1D arrays, and prepares the labels for the machine learning model. The processed images and labels are then saved to a file for later use in training.
| `predict_webcam.py` | Provided code that uses the trained model to recognize objects in real time. |


---

### 1. Install requirements

```bash

pip install -r requirements.txt


```

### 2. Collect training images
Use the webcam to take pictures of classroom objects:

```bash
python save_images.py
```
Create 20–30 images per object (pen, glasses, etc.).

### 3. Train your model
```bash
python train_model.py
```
This script will process the images and train a model called object_detector.h5.

### 4. Run real-time prediction!
```bash
python predict_webcam.py
```

Try holding up different objects in front of the webcam. The program will tell you what it sees!

### Clean-Up Tip
If the model isn’t working well, you can delete the object_detector.h5 file and try collecting more images or re-training.

### Learning Goals
 - Understand how machines can learn to see like humans
 - Learn basic Python scripting and libraries like OpenCV and TensorFlow
 - Experience the joy of coding and seeing results in real-time!

### During the Show & Tell
Encourage students to:
 - Show live object recognition
 - Explain how they trained their model
 - Talk about challenges and what they’d like to try next!
