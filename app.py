import os
import csv
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from utils.ans import process_images_and_compare
from utils.pdf_splitter import split_pdf

app = Flask(__name__ )

# Folder configuration
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
RESULTS_FOLDER = './results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', upload_url='/upload')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files or 'answer_key' not in request.files:
        return jsonify({'error': 'No files or answer key provided.'}), 400

    files = request.files.getlist('files[]')
    answer_key_file = request.files['answer_key']
    if not files:
        return jsonify({'error': 'No PDF files uploaded.'}), 400

    # Save answer key
    answer_key_filename = secure_filename(answer_key_file.filename)
    answer_key_path = os.path.join(UPLOAD_FOLDER, answer_key_filename)
    answer_key_file.save(answer_key_path)

    results = []
    for file in files:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(pdf_path)

            # Split PDF into images
            image_output_folder = os.path.join(PROCESSED_FOLDER, filename.rsplit('.', 1)[0])
            os.makedirs(image_output_folder, exist_ok=True)
            split_pdf(pdf_path, image_output_folder)

            # Process images and compare with answer key
            image_results = process_images_and_compare(image_output_folder, answer_key_path)
            results.extend(image_results)
        else:
            return jsonify({'error': f'Invalid file type: {file.filename}.'}), 400

    # Save results to CSV
    
    csv_path = os.path.join(RESULTS_FOLDER, 'results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Page', 'Enrollment Number', 'Image', 'Predicted Answers', 'Correctness', 'Answer Key', 'Total Score'])
        for result in results:
            for i in range(len(result['predicted_answers'])):
                writer.writerow([
                    result['page'],
                    result['enrollment'],  # Add enrollment number to CSV
                    result['image'],
                    result['predicted_answers'][i],
                    '✅' if result['correctness'][i] else '❌',
                    result['answer_key'][i],
                    result['total_score']
            ])

    return render_template('result.html', results=results, csv_path='results.csv')

@app.route('/download_csv')
def download_csv():
    csv_path = os.path.join(RESULTS_FOLDER, 'results.csv')
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True)
    else:
        return jsonify({'error': 'CSV not found.'}), 404

if __name__ == '__main__':
    app.run(debug=True)