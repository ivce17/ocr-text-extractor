import cv2
import pytesseract
import os
import numpy as np
from PIL import Image

# If you installed to the default location, it should look exactly like this:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Folders based on your project structure
input_folder = './images_raw'
processed_folder = './images_processed'
output_text_file = 'final_digitized_book.txt'

# Create processed folder if it doesn't exist
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)


def deskew(image):
    """Straightens the image if it was scanned at an angle."""
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_for_ocr(image_path, filename):
    # Load image
    img = cv2.imread(image_path)

    # A. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # B. Denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # C. Thresholding (Binarization)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # D. Deskewing
    final_img = deskew(thresh)

    # Save processed image for your project records
    cv2.imwrite(os.path.join(processed_folder, f"proc_{filename}"), final_img)

    return final_img


# 2. EXECUTION
images = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

with open(output_text_file, 'w', encoding='utf-8') as f:
    for filename in images:
        print(f"Processing: {filename}...")
        full_path = os.path.join(input_folder, filename)

        # Run Pipeline
        processed_img = preprocess_for_ocr(full_path, filename)

        # Run OCR
        text = pytesseract.image_to_string(processed_img, lang='mkd', config='--psm 3')

        # Write to file
        f.write(f"\n\n--- СТРАНИЦА: {filename} ---\n\n")
        f.write(text)

print(f"\nSuccess! Check '{processed_folder}' for cleaned images and '{output_text_file}' for the text.")