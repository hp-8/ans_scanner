import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path

def get_next_index(output_folder):
    """Get the next index to avoid overwriting existing images."""
    existing_files = os.listdir(output_folder)
    existing_indices = []
    for file in existing_files:
        if file.endswith(".jpg") or file.endswith(".png"):
            try:
                # Extract the number from the filename (e.g., page_001.jpg -> 001)
                index = int(file.split('_')[-1].split('.')[0])
                existing_indices.append(index)
            except ValueError:
                continue
    # Return the next available index
    return max(existing_indices, default=0) + 1

def split_pdf(pdf_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract the file name from the path to handle output naming
    base_filename = os.path.basename(pdf_path).split('.')[0]

    # Get the next available index to prevent overwriting files
    next_index = get_next_index(output_folder)

    # Load the PDF
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    # Convert PDF pages to images
    images = convert_from_path(pdf_path)

    for page_number, image in enumerate(images):
        image_filename = f"{base_filename}_page_{next_index:03d}.jpg"
        image_path = os.path.join(output_folder, image_filename)

        # Save the image
        image.save(image_path, 'JPEG')
        print(f"Saved {image_filename}")

        next_index += 1  # Increment the index for the next image
# Usage example (this can be commented out when you import this in app.py)
# split_pdf("/Users/hp_8/Desktop/work/ans_scanner/data/pdfs/id_train_9.pdf", "/Users/hp_8/Desktop/work/ans_scanner/data/images/val")