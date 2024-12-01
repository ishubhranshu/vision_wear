from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import faiss
import numpy as np
import base64
from io import BytesIO

# Import utility functions
from utils.encoding import (
    encode_class,
    encode_timestamp,
    encode_bounding_box,
    create_feature_vector
)
from utils.processing import (
    process_video,
    create_annotated_video
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key in production
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ANNOTATED_FOLDER'] = "static/videos"
app.config['FRAMES_FOLDER'] = "static/frames"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATED_FOLDER'], exist_ok=True)
os.makedirs(app.config['FRAMES_FOLDER'], exist_ok=True)

# Load YOLO model
from ultralytics import YOLO
model = YOLO("models/best.pt")  # Ensure the model path is correct

# Initialize Sentence Transformer for class name embeddings
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient

# Initialize FAISS index
feature_dim = 384 + 1 + 4  # class embedding (384) + timestamp (1) + bounding box (4) = 389
faiss_index = faiss.IndexFlatL2(feature_dim)

# Initialize detection metadata list
detection_metadata = []

# Global variables to store application state
uploaded_video_path = None
class_counts = {}
class_colors = {}
processed_results = []
selected_classes = []
annotated_video_filename = None
total_detections = 0

# Allowed File Extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    """
    Checks if the uploaded file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home Route
@app.route("/", methods=["GET", "POST"])
def home():
    global uploaded_video_path, class_counts, class_colors, processed_results, selected_classes, annotated_video_filename, total_detections, detection_metadata

    if request.method == "POST":
        # Handle video upload
        if "video" in request.files:
            video = request.files["video"]
            if video.filename == '':
                flash("No selected file")
                return redirect(request.url)
            if not allowed_file(video.filename):
                flash("Unsupported file type. Please upload a valid video file.")
                return redirect(request.url)
            filename = secure_filename(video.filename)
            uploaded_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(uploaded_video_path)

            # Process video to extract classes and counts
            class_counts, processed_results, detection_metadata = process_video(
                uploaded_video_path,
                model,
                embedding_model,
                faiss_index,
                app.config['FRAMES_FOLDER']
            )
            total_detections = sum(class_counts.values())
            print(f"Detected classes: {class_counts}")
            selected_classes = []
            annotated_video_filename = None
            return redirect(url_for("home"))

        # Handle class selection via checkboxes
        elif "classes" in request.form:
            selected_classes = request.form.getlist("classes")
            print(f"Selected classes: {selected_classes}")
            if not selected_classes:
                flash("No classes selected.")
                return redirect(url_for("home"))
            # Process video to create annotated video
            original_video_basename = os.path.splitext(os.path.basename(uploaded_video_path))[0]
            selected_classes_sorted = sorted(selected_classes)  # Sort for consistent filename
            selected_classes_str = '_'.join(selected_classes_sorted)
            annotated_video_filename = f"annotated_{selected_classes_str}_{original_video_basename}.mp4"
            annotated_video_path = os.path.join(app.config['ANNOTATED_FOLDER'], annotated_video_filename)
            create_annotated_video(
                uploaded_video_path,
                processed_results,
                selected_classes,
                annotated_video_path,
                class_colors
            )
            return redirect(url_for("home"))

    return render_template(
        "main.html",
        uploaded_video_path=uploaded_video_path,
        class_counts=class_counts,
        selected_classes=selected_classes,
        annotated_video_filename=annotated_video_filename,
        total_detections=total_detections
    )

# Reset Route
@app.route("/reset")
def reset():
    """
    Reset the global variables to their initial state and redirect to home.
    """
    global uploaded_video_path, class_counts, class_colors, processed_results, selected_classes, annotated_video_filename, total_detections, detection_metadata
    uploaded_video_path = None
    class_counts = {}
    class_colors = {}
    processed_results = []
    selected_classes = []
    annotated_video_filename = None
    total_detections = 0
    detection_metadata = []
    return redirect(url_for("home"))

# Search Route
@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        # Get query parameters
        query_class = request.form.get("query_class")
        query_timestamp = request.form.get("query_timestamp")
        query_bbox = request.form.get("query_bbox")

        # Validate inputs
        if not query_class or not query_timestamp or not query_bbox:
            flash("All fields are required.")
            return redirect(url_for("search"))

        try:
            query_timestamp = float(query_timestamp)
            query_bbox = list(map(float, query_bbox.split(',')))
            if len(query_bbox) != 4:
                raise ValueError
        except ValueError:
            flash("Invalid input for timestamp or bounding box.")
            return redirect(url_for("search"))

        # Get video properties (assuming single video; adjust if multiple)
        if uploaded_video_path:
            import cv2
            cap = cv2.VideoCapture(uploaded_video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = frame_count / fps
            cap.release()
        else:
            flash("No video uploaded.")
            return redirect(url_for("search"))

        # Encode the query
        query_feature = create_feature_vector(
            query_class, query_timestamp, query_bbox, video_duration, frame_width, frame_height
        )

        # Perform FAISS search
        k = 5  # Number of nearest neighbors
        D, I = faiss_index.search(np.expand_dims(query_feature, axis=0), k)  # D: distances, I: indices

        # Retrieve metadata for the top k results
        results = []
        for idx in I[0]:
            if idx < len(detection_metadata):
                res = detection_metadata[idx].copy()  # Make a copy to avoid mutation
                # Load the corresponding frame image
                frame_number = res["frame_number"]
                frame_filename = f"frame_{frame_number}.jpg"
                frame_path = os.path.join(app.config['FRAMES_FOLDER'], frame_filename)
                if os.path.exists(frame_path):
                    # Load image using OpenCV
                    image = cv2.imread(frame_path)
                    if image is not None:
                        # Draw the bounding box
                        x1, y1, x2, y2 = map(int, res["bbox"])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color in BGR
                        cv2.putText(image, res["class"], (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        # Convert image to RGB
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # Encode image to JPEG in memory
                        retval, buffer = cv2.imencode('.jpg', image_rgb)
                        if retval:
                            img_bytes = buffer.tobytes()
                            # Encode to base64
                            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                            # Create data URI
                            img_uri = f"data:image/jpeg;base64,{img_base64}"
                            # Add image URI to result
                            res["image"] = img_uri
                        else:
                            res["image"] = None
                    else:
                        res["image"] = None
                else:
                    res["image"] = None
                results.append(res)

        return render_template("search_results.html", results=results)

    return render_template("search.html")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, port=5000)
