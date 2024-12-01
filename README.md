# YOLO Detection with FAISS Integration

![Project Logo](logo.png) <!-- Replace with your project logo -->
<a href="https://www.flaticon.com/free-icons/clothing" title="clothing icons">Clothing icons created by Freepik - Flaticon</a>

## Introduction

Welcome to the **YOLO Detection with FAISS Integration** project! This Flask-based web application empowers users to upload videos, perform real-time object detection using the YOLO (You Only Look Once) model, and efficiently search through detections using FAISS (Facebook AI Similarity Search). Whether you're analyzing surveillance footage, creating annotated videos, or conducting research, this tool provides a robust and user-friendly interface to streamline your workflow.

## Features

- **Video Upload**: Seamlessly upload videos in popular formats such as MP4, AVI, MOV, and MKV.
- **Real-Time Object Detection**: Utilize the powerful YOLO model to detect and categorize objects within each frame of the uploaded video.
- **Feature Encoding**: Encode detection features using Sentence Transformers for enhanced similarity search capabilities.
- **Efficient Similarity Search**: Implement FAISS to index and search through detections, enabling quick retrieval of similar objects based on class name, timestamp, and bounding box.
- **Annotated Video Generation**: Generate and view annotated videos that highlight selected object classes, making it easier to focus on areas of interest.
- **User-Friendly Interface**: Intuitive web interface designed for ease of use, ensuring a smooth user experience even for those without technical expertise.
- **Search Functionality**: Advanced search options to query detections based on specific criteria, enhancing data analysis and exploration.

## Demo

![Upload Video](static/images/demo_upload.png)  
*Upload your video to get started.*

![Select Classes](static/images/demo_select_classes.png)  
*Choose the classes you want to highlight in the annotated video.*

![Annotated Video](static/images/demo_annotated_video.png)  
*View the generated annotated video with highlighted objects.*

![Search Detections](static/images/demo_search.png)  
*Search for specific detections using class name, timestamp, and bounding box.*

![Search Results](static/images/demo_search_results.png)  
*Review the search results with detailed information and frame snapshots.*

> **Note**: Ensure that the image paths (`static/images/demo_*.png`) correspond to your actual screenshot filenames.

## File Description

Here's a brief overview of the project's directory structure and key files:

