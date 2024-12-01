# utils/processing.py

import os
import cv2
from ultralytics import YOLO
from moviepy.editor import ImageSequenceClip
import random
import faiss
import numpy as np
from utils.encoding import create_feature_vector

def process_video(video_path, model, embedding_model, faiss_index, frames_dir):
    """
    Processes the uploaded video to detect objects, encode features, save frames, and index them with FAISS.
    Returns class counts, processed results, and detection metadata.
    """
    class_counts = {}
    processed_results = []
    class_colors = {}
    detection_metadata = []  # Reset metadata

    os.makedirs(frames_dir, exist_ok=True)

    # Function to generate a unique color for each class
    def get_unique_color(existing_colors):
        while True:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if color not in existing_colors.values():
                return color

    # Obtain video properties
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps  # in seconds

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image
        frame_filename = f"frame_{frame_number}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Perform detection
        results = model(frame)
        frame_results = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                # Assign a unique color to each class if not already assigned
                if cls_name not in class_colors:
                    class_colors[cls_name] = get_unique_color(class_colors)
                # Update counts
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                # Get bounding box coordinates
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                frame_results.append({"class": cls_name, "bbox": bbox})

                # Encode the detection
                timestamp = frame_number / fps  # Time in seconds
                feature_vector = create_feature_vector(
                    cls_name, timestamp, bbox, video_duration, frame_width, frame_height
                )
                # Add to FAISS index
                faiss_index.add(np.expand_dims(feature_vector, axis=0))
                # Store metadata
                detection_metadata.append({
                    "class": cls_name,
                    "timestamp": round(timestamp, 2),
                    "bbox": [round(coord, 2) for coord in bbox],
                    "frame_number": frame_number
                })

        processed_results.append(frame_results)
        frame_number += 1

    cap.release()
    return class_counts, processed_results, detection_metadata

def create_annotated_video(original_video_path, processed_results, selected_classes, annotated_video_path, class_colors):
    """
    Creates an annotated video highlighting only the selected classes.
    """
    cap = cv2.VideoCapture(original_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_index >= len(processed_results):
            break

        if frame_index < len(processed_results):
            for detection in processed_results[frame_index]:
                cls_name = detection["class"]
                bbox = detection["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                color = class_colors.get(cls_name, (255, 0, 0))  # Default to red if color not found
                # Draw bounding boxes only for selected classes
                if cls_name in selected_classes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, cls_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert frame to RGB (moviepy uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_index += 1

    cap.release()

    if frames:
        # Use moviepy to create video
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(annotated_video_path, codec='libx264', audio=False, verbose=False, logger=None)
    else:
        print("No frames to write for annotated video.")
