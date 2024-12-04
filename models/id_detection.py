import pytesseract
import cv2

def detect_student_id(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    student_id = pytesseract.image_to_string(gray, config="--psm 6 digits")
    return student_id.strip() or "UNKNOWN"