# train.py
from ultralytics import YOLO

# Load YOLOv8 model (you can use a pre-trained model like 'yolov8n.pt' or 'yolov8s.pt' for better performance)
model = YOLO('yolov8n.pt')  # You can use yolov8s or yolov8m for better performance

# Train the model with the dataset.yaml file that contains dataset details
model.train(
    data='dataset.yaml',    # Path to the dataset.yaml file
    epochs=50,              # Number of epochs, adjust based on your needs
    imgsz=640,              # Image size for training
    batch=16,               # Batch size, adjust based on your GPU capabilities
    name='ans_scanner_model',  # Name of the model training folder
    exist_ok=True           # If a folder with the same name exists, it will overwrite it
)