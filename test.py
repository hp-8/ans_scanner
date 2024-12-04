import os
import argparse
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Paths
TEST_IMAGES_DIR = "/Users/hp_8/Desktop/work/ans_scanner/data/images/test"  # Directory with test images
TEST_LABELS_DIR = "/Users/hp_8/Desktop/work/ans_scanner/data/labels/test"  # Directory with test annotations

def load_ground_truth(label_path):
    """
    Load ground truth annotations from a YOLO label file.
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Parse labels into bounding box coordinates and class IDs
    gt_boxes = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        bbox = [float(p) for p in parts[1:]]
        gt_boxes.append((class_id, bbox))  # Class ID and bounding box

    return gt_boxes

def evaluate_model(model_path, test_images_dir, test_labels_dir):
    """
    Evaluate a YOLO model on a test dataset.
    """
    # Load the trained model
    model = YOLO(model_path)

    all_gt_classes = []
    all_pred_classes = []

    # Iterate through test images
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        label_path = os.path.join(test_labels_dir, image_name.replace(".jpg", ".txt"))

        # Skip if label file doesn't exist
        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {image_name}")
            continue

        # Load ground truth labels
        ground_truth = load_ground_truth(label_path)
        gt_classes = [gt[0] for gt in ground_truth]

        # Run inference on the image
        results = model(image_path)
        detections = results[0].boxes.data  # YOLO predictions
        
        # Parse predicted classes
        pred_classes = [int(detection[-1].item()) for detection in detections]

        # Aggregate results
        all_gt_classes.extend(gt_classes)
        all_pred_classes.extend(pred_classes)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_gt_classes, all_pred_classes, average="weighted")
    report = classification_report(all_gt_classes, all_pred_classes, target_names=model.names)

    print("\nModel Evaluation:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on test data.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained YOLO model file (.pt).")
    parser.add_argument("--images", type=str, default=TEST_IMAGES_DIR, help="Path to test images directory.")
    parser.add_argument("--labels", type=str, default=TEST_LABELS_DIR, help="Path to test labels directory.")
    args = parser.parse_args()

    evaluate_model(args.model, args.images, args.labels)