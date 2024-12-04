import torch

def detect_answers(image_path):
    # Load YOLOv8 model
    model = torch.hub.load('ultralytics/ultralytics', 'yolov8', pretrained=True)
    
    # Run the model on the image
    results = model(image_path)
    
    # Debug: Print results to inspect the structure
    print(results)
    print(results.boxes)  # This contains bounding box information
    
    # Extract detections
    detections = results.boxes  # YOLOv8 uses `.boxes`
    
    answers = []
    for box in detections:
        # Each box contains x1, y1, x2, y2, confidence, class
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = box.cls[0].item()
        
        answers.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'class': cls
        })
    
    return answers