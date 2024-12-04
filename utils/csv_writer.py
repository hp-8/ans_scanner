import csv

def generate_results_csv(results, answer_key, output_path):
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Student ID", "Score", "Correct", "Answers"])
        for student_id, answers in results:
            correct = sum(1 for i, ans in enumerate(answers) if ans == answer_key[f"Q{i+1}"])
            score = round((correct / len(answer_key)) * 100, 1)
            writer.writerow([student_id, score, correct, answers])