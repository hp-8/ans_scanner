from ultralytics import YOLO

# Load the YOLOv8 model during import
model = YOLO('/Users/hp_8/Desktop/work/ans_scanner/yolov8n.pt')

def detect_answers(image_path):
    """
    Detect objects (answers) in the given image using YOLOv8.
    Args:
        image_path (str): Path to the input image.

    Returns:
        list: Parsed detection results.
    """
    # Perform inference
    results = model(image_path)
    
    # Parse detections
    parsed_results = []
    for detection in results[0].boxes.data:
        x_min, y_min, x_max, y_max, confidence, class_id = detection.tolist()
        parsed_results.append({
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'confidence': confidence,
            'class_id': int(class_id)  # Convert to integer
        })
    return parsed_results

counter = 1

def detect_student_Id():
    global counter
    if counter > 15:
        counter = 1
    enrollment_number = f"2024390{counter:02d}"
    counter += 1
    return enrollment_number
