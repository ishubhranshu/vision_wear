# utils/encoding.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize Sentence Transformer for class name embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient

def encode_class(class_name):
    """
    Encodes the class name into a dense vector using Sentence Transformers.
    """
    embedding = embedding_model.encode(class_name)
    return embedding  # Shape: (384,) for 'all-MiniLM-L6-v2'

def encode_timestamp(timestamp, video_duration):
    """
    Normalizes the timestamp to a value between 0 and 1.
    """
    return np.array([timestamp / video_duration], dtype=np.float32)

def encode_bounding_box(bbox, frame_width, frame_height):
    """
    Normalizes the bounding box coordinates.
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    return np.array([
        x1 / frame_width,
        y1 / frame_height,
        x2 / frame_width,
        y2 / frame_height
    ], dtype=np.float32)

def create_feature_vector(class_name, timestamp, bbox, video_duration, frame_width, frame_height):
    """
    Creates a combined feature vector for FAISS indexing.
    """
    class_emb = encode_class(class_name)  # Shape: (384,)
    timestamp_enc = encode_timestamp(timestamp, video_duration)  # Shape: (1,)
    bbox_enc = encode_bounding_box(bbox, frame_width, frame_height)  # Shape: (4,)
    feature_vector = np.concatenate([class_emb, timestamp_enc, bbox_enc])  # Shape: (389,)
    return feature_vector.astype('float32')  # FAISS requires float32
