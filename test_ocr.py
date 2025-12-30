import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

# Set Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def test_depth_view(video_path='fish.mp4'):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Convert to PIL to match backend logic
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size

        # --- ADJUST THIS MULTIPLIER ---
        # 0.88 targets the right-most 12%. 
        # Try 0.85 if the numbers are being cut off on the left.
        left, top, right, bottom = 0, 100, int(w * 0.25), int(h * 0.11)
        roi_depth = frame_pil.crop((left, top, right, bottom))
        
        # Pre-process for OCR
        roi_cv = cv2.cvtColor(np.array(roi_depth), cv2.COLOR_RGB2GRAY)
        # Scale up by 2x for better digit recognition
        roi_cv = cv2.resize(roi_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Thresholding to get clear black text on white background
        _, thresh = cv2.threshold(roi_cv, 180, 255, cv2.THRESH_BINARY_INV)

        # SHOW THE PROCESSED REGION
        cv2.imshow('Processed Depth Region', thresh)
        
        # Run OCR
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        # Extract numbers like 0.8, 0.9
        depth_matches = re.findall(r"(\d+\.\d+)", text)
        
        print("-" * 30)
        print(f"Raw OCR Text: {text.strip()}")
        print(f"Extracted Depths: {depth_matches}")
        print("-" * 30)
        print("Press any key on the image window to close.")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change this to your actual image file name
    test_depth_view(r'C:\Users\HP\Documents\FYP DATASET\fyp_test\fish.mp4')