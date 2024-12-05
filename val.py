import os
import csv
from ultralytics import YOLO  # For YOLOv5 or YOLOv8
import pandas as pd

# Load the trained YOLO model
model = YOLO('/Users/hp_8/Desktop/work/ans_scanner/runs/detect/train11/weights/best.pt')  # Path to your trained model weights

# Path to the images you want to predict
images_folder = '/Users/hp_8/Desktop/work/ans_scanner/data/images/test'

# Run the prediction
results = model.predict(source=images_folder, conf=0.25, save_txt=False)

# Initialize an array to store the result data
result_data = []

# Process the results for each image
for result in results:
    for idx, (boxes, confs, class_ids) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
        for i in range(len(boxes)):
            # Ensure class_id is extracted properly as an integer
            class_id = int(class_ids[i].item())  # Convert tensor to integer

            # Prepare the results for each detected object
            result_data.append({
                'Image': result.path,
                'Class': model.names[class_id],  # Get the class name using the class id
                'Confidence': confs[i],
                'Bounding Box': boxes[i].tolist()  # Convert to list for easy storage
            })

# Convert the result data to a DataFrame (Tabular format)
df = pd.DataFrame(result_data)

# Save the result as a CSV file (ensure the path is correct)
csv_file = '/Users/hp_8/Desktop/work/ans_scanner/results/predictions.csv'

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

df.to_csv(csv_file, index=False)

print(f"Results saved to {csv_file}")