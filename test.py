import os
import argparse
from ultralytics import YOLO

# Paths
DEFAULT_IMAGES_DIR = "/Users/hp_8/Desktop/work/ans_scanner/data/images/test"  # Directory with test images

def test_model_without_ground_truth(model_path, test_images_dir):
    """
    Test a YOLO model on a set of images and output predictions.
    """
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Iterate through test images
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)

        # Skip non-image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {image_name}")
            continue

        # Run inference on the image
        print(f"\nProcessing image: {image_name}")
        results = model(image_path)
        
        # Parse predictions
        detections = results[0].boxes.data if results[0].boxes is not None else []
        for detection in detections:
            x_min, y_min, x_max, y_max, confidence, class_id = detection.tolist()
            print(f"Class: {int(class_id)}, Confidence: {confidence:.2f}, "
                  f"Box: ({x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a YOLO model on images without ground truth.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained YOLO model file (.pt).")
    parser.add_argument("--images", type=str, default=DEFAULT_IMAGES_DIR, help="Path to test images directory.")
    args = parser.parse_args()

    test_model_without_ground_truth(args.model, args.images)