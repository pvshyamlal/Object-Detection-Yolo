import torch
import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# Helper function to plot bounding boxes
def plot_one_box(xyxy, img, color=(0, 255, 0), label=None, line_thickness=3):
    """Plots one bounding box on the image."""
    tl = line_thickness  # line thickness
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=tl)  # Rectangle
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_size = cv2.getTextSize(label, font, 0.5, 1)[0]
        cv2.putText(img, label, (x1, y1 - 10), font, 0.5, color, 1, cv2.LINE_AA)

def run_yolo(input_folder, output_folder, model_weights='yolov5s.pt', conf_threshold=0.25):
    # Ensure input and output folders exist
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Check if the input folder exists
    if not input_folder.exists():
        print(f"Error: The input folder '{input_folder}' does not exist.")
        return

    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Process each image in the input folder
    for image_path in input_folder.glob('*.*'):
        print(f"Processing {image_path.name}...")

        # Read the image using OpenCV
        img = cv2.imread(str(image_path))

        # Convert image to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference on the image
        results = model(img_rgb)

        # Process detections
        results.render()  # Render bounding boxes on the image

        # Save the result image
        output_image_path = output_folder / image_path.name
        cv2.imwrite(str(output_image_path), results.ims[0])  # Save the result image

        print(f"Saved result: {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True, help='Path to input images folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to save output images')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Path to YOLOv5 weights')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detection')

    opt = parser.parse_args()

    run_yolo(opt.input_folder, opt.output_folder, opt.weights, opt.conf_threshold)
