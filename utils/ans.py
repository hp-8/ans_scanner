import os
from ultralytics import YOLO
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)

# Load the YOLO model
model = YOLO('/Users/hp_8/Desktop/work/ans_scanner/runs/detect/train11/weights/best.pt')  # Replace with your model path

def predict_labels(image_path):
    """
    Predict labels for a given image.
    :param image_path: Path to the image.
    :return: List of detected answers.
    """
    results = model.predict(image_path, conf=0.25, iou=0.45)
    detected_answers = []
    for result in results:
        detected_classes = [model.names[int(class_id)] for class_id in result.boxes.cls]
        detected_answers.extend(detected_classes)
    return detected_answers

def parse_answer_key(answer_key_path):
    """
    Parse the answer key from a text file.
    :param answer_key_path: Path to the answer key file.
    :return: Dictionary of question number to correct answer.
    """
    answer_key = {}
    with open(answer_key_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            answer_key[i + 1] = line.strip().lower()  # Case-insensitive
    return answer_key

def compare_answers(predictions, answer_key):
    """
    Compare predicted answers with the answer key.
    :param predictions: List of predicted answers.
    :param answer_key: Dictionary of correct answers.
    :return: List of correctness (True/False).
    """
    correctness = []
    for i, predicted_answer in enumerate(predictions):
        correct_answer = answer_key.get(i + 1, '')
        correctness.append(predicted_answer.lower() == correct_answer)  # Case-insensitive comparison
    return correctness

def process_images_and_compare(images_folder, answer_key_path):
    """
    Process images, predict labels, and compare with the answer key.
    :param images_folder: Folder containing the images.
    :param answer_key_path: Path to the answer key file.
    :return: List of results.
    """
    answer_key = parse_answer_key(answer_key_path)
    results = []

    for image in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image)
        if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            predicted_answers = predict_labels(image_path)
            correctness = compare_answers(predicted_answers, answer_key)
            results.append({
                'image': image,
                'predicted_answers': predicted_answers,
                'correctness': correctness,
                'answer_key': [answer_key.get(i + 1, '') for i in range(len(predicted_answers))]
            })
    return results