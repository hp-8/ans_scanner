import os
from ultralytics import YOLO
import logging
import random
import math
from models.yolo_model import detect_student_Id

logging.basicConfig(level=logging.INFO)

model = YOLO('/Users/hp_8/Desktop/work/ans_scanner/runs/detect/train11/weights/best.pt')

def predict_labels(image_path, answer_key=None):
    results = model.predict(image_path, conf=0.25, iou=0.45)
    detected_answers = []

    for result in results:
        detected_classes = [model.names[int(class_id)] for class_id in result.boxes.cls]
        detected_answers.extend(detected_classes)

    if answer_key:
        for i, answer in enumerate(detected_answers):
            question_number = i + 1
            correct_answer = answer_key.get(question_number, '').lower()
            if answer.lower() == correct_answer:
                detected_answers[i] = answer.upper()
            else:
                detected_answers[i] = answer.lower()

    return detected_answers

def parse_answer_key(answer_key_path):
    answer_key = {}
    with open(answer_key_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            answer_key[i + 1] = line.strip().lower()
    return answer_key

def compare_answers(predictions, answer_key):
    correctness = []
    formatted_answers = []

    for i, predicted_answer in enumerate(predictions):
        correct_answer = answer_key.get(i + 1, '').lower()
        is_correct = predicted_answer.lower() == correct_answer
        correctness.append(is_correct)
        if is_correct:
            formatted_answers.append(predicted_answer.upper())
        else:
            formatted_answers.append(predicted_answer.lower())

    return formatted_answers, correctness

def process_images_and_compare(images_folder, answer_key_path):
    answer_key = parse_answer_key(answer_key_path)
    results = []

    page_number = 1
    for image in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image)
        if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            predicted_answers = predict_labels(image_path)
            formatted_answers, correctness = compare_answers(predicted_answers, answer_key)
            correct_count = sum(correctness)
            total_questions = 100
            total_score = math.ceil((correct_count / total_questions) * 100)
            enrollment_number = detect_student_Id()

            results.append({
                'page': page_number,
                'image': image,
                'enrollment': enrollment_number,
                'predicted_answers': formatted_answers,
                'correctness': correctness,
                'answer_key': [answer_key.get(i + 1, '') for i in range(len(predicted_answers))],
                'total_score': total_score
            })
            page_number += 1

    return results