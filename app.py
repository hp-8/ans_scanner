import os
import csv
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from utils.pdf_splitter import split_pdf
from models.yolo_model import detect_answers  # Updated YOLO model import

app = Flask(__name__)

# Folder configuration
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
RESULTS_FOLDER = './results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Configure upload size limit
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Allowed extensions
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', upload_url='/upload')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided. Please upload at least one PDF file.'}), 400

    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No files uploaded. Please select valid files.'}), 400

    results = []
    for file in files:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(pdf_path)

            # Split the PDF into images
            image_output_folder = os.path.join(PROCESSED_FOLDER, filename.rsplit('.', 1)[0])
            os.makedirs(image_output_folder, exist_ok=True)
            split_pdf(pdf_path, image_output_folder)

            # Process each image with the model
            images = [os.path.join(image_output_folder, img) for img in os.listdir(image_output_folder)]
            for img in images:
                try:
                    detections = detect_answers(img)  # Run the model
                    correct_answers = sum(1 for det in detections if det['confidence'] > 0.5)
                    total_questions = len(detections)
                    marks = (correct_answers / total_questions) * 100 if total_questions > 0 else 0

                    results.append({
                        "Filename": filename,
                        "Page": os.path.basename(img),
                        "Correct Answers": correct_answers,
                        "Total Questions": total_questions,
                        "Marks": round(marks, 2),
                        "Detections": detections
                    })
                except Exception as e:
                    results.append({
                        "Filename": filename,
                        "Page": os.path.basename(img),
                        "Error": str(e)
                    })
        else:
            return jsonify({'error': f'Invalid file type: {file.filename}. Only PDFs are allowed.'}), 400

    # Save results to CSV
    csv_file = os.path.join(RESULTS_FOLDER, "final_results.csv")
    with open(csv_file, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Filename", "Page", "Correct Answers", "Total Questions", "Marks"])
        writer.writeheader()
        for result in results:
            writer.writerow({
                "Filename": result["Filename"],
                "Page": result["Page"],
                "Correct Answers": result.get("Correct Answers", "N/A"),
                "Total Questions": result.get("Total Questions", "N/A"),
                "Marks": result.get("Marks", "N/A")
            })

    return render_template("results.html", results=results, csv_file="final_results.csv")

@app.route('/download_csv')
def download_csv():
    csv_path = os.path.join(RESULTS_FOLDER, "final_results.csv")
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True)
    else:
        return jsonify({'error': 'CSV file not found.'}), 404

if __name__ == '__main__':
    app.run(debug=True)