import subprocess
import ctypes
import win32gui
import win32ui
import pygetwindow as gw
import numpy as np
import time
import cv2
import joblib
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from config import scrcpy_path, scrcpy_title, model_paths
from anomaly_detector import AnomalyDetector
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust if installed elsewhere
import easyocr

class DetectionBackend:
    def __init__(self):
        self.model = None
        self.current_detection_type = None
        self.scrcpy_process = None
        self.hwnd = None
        self.video_cap = None
        self.running_source = None  # "phone" or "video"
        self.anomaly_detector = AnomalyDetector()
        self.fish_model = None
        self._load_fish_model()
        self.reader = easyocr.Reader(['en'], gpu=False)  # Initialize EasyOCR reader
    
    def _load_fish_model(self):
        """Load the fish regression model"""
        try:
            self.fish_model = joblib.load(r'sonar_model_randomforest.joblib')
            print("Fish regression model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load fish model: {e}")
            self.fish_model = None

    
    def load_model(self, detection_type):
        """Load YOLO model for specified detection type"""
        if detection_type != self.current_detection_type or self.model is None:
            print(f"Loading {detection_type} model...")
            try:
                self.model = YOLO(model_paths[detection_type])
                self.current_detection_type = detection_type
                print(f"{detection_type} model loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return True
    
    def capture_window(self, hwnd):
        """Capture scrcpy window"""
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        w, h = right - left, bottom - top
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        save_bit = win32ui.CreateBitmap()
        save_bit.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(save_bit)
        ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 1)
        bmpinfo = save_bit.GetInfo()
        bmpstr = save_bit.GetBitmapBits(True)
        img = Image.frombuffer("RGB", (bmpinfo["bmWidth"], bmpinfo["bmHeight"]), bmpstr, "raw", "BGRX", 0, 1)
        win32gui.DeleteObject(save_bit.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        return img
    
    # def _extract_depth(self, frame):
        """Extract depth meter reading from the right side of the sonar"""
        try:
            # 1. Define ROI (same as test_ocr.py)
            w, h = frame.size
            left, top, right, bottom = 0, 100, int(w * 0.25), int(h * 0.11)
            roi_depth = frame.crop((left, top, right, bottom))
            
            # 2. Pre-process (match test_ocr.py exactly: grayscale, resize 2x, threshold 180)
            roi_cv = cv2.cvtColor(np.array(roi_depth), cv2.COLOR_RGB2GRAY)
            roi_cv = cv2.resize(roi_cv, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)  # Add resize
            roi_cv = cv2.GaussianBlur(roi_cv, (3, 3), 0)  # Add blur
            _, thresh = cv2.threshold(roi_cv, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)  # Match threshold
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.dilate(thresh, kernel,iterations=1)  # Add dilation
            
            # 3. Perform OCR (match test_ocr.py config: no 'm' in whitelist)
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.m'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            
            # Debug: Print OCR text to compare with test_ocr.py
            print(f"Backend OCR Text: '{text.strip()}'")
            
            # 4. Extract numbers (same regex)
            depth_matches = re.findall(r"(\d+\.\d+)", text)
            
            return depth_matches[0] if depth_matches else "Unknown"
        except Exception as e:
            print(f"Depth extraction error: {e}")
            return "N/A"
    def _extract_depth(self, frame):
        """Extract depth using EasyOCR for higher power and accuracy"""
        try:
            import re
            w, h = frame.size
            
            # 1. Target the top-left box (Precise ROI)
            if self.running_source == "video":
                left, top, right, bottom = 0, 100, int(w * 0.25), int(h * 0.11)
                roi_depth = frame.crop((left, top, right, bottom))
                # 2. Convert to Grayscale (EasyOCR works well with simple images)
                roi_cv = cv2.cvtColor(np.array(roi_depth), cv2.COLOR_RGB2GRAY)
            else:
                left = 0
                top = 60       # Try changing this if you see the battery icon
                width = int(w * 0.30)  # 40% width to catch the text
                height = 53  # Tall box to ensure we don't miss it
                            
                right = left + width
                bottom = top + height
                roi_depth = frame.crop((left, top, right, bottom))
                # 2. PRE-PROCESSING (The Fix for 'xm')
                # Convert to numpy array
                roi_np = np.array(roi_depth)
                roi_cv = cv2.cvtColor(roi_np, cv2.COLOR_RGB2GRAY)
                
                # A. UPSCALE: Make the text 3x bigger so EasyOCR can see the shape
                roi_cv = cv2.resize(roi_cv, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                
                # B. SHARPEN: Apply thresholding to force text to be Black & White
                # This removes the "compression fuzz" from scrcpy
                _, roi_cv = cv2.threshold(roi_cv, 120, 255, cv2.THRESH_BINARY)
                
                # Optional: Add white border padding (helps OCR read edge characters)
                roi_cv = cv2.copyMakeBorder(roi_cv, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255])
            # 3. Use EasyOCR to read the text
            # detail=0 returns only the detected text string
            results = self.reader.readtext(roi_cv, detail=0)
            
            if not results: return "Unknown"
            raw_text = "".join(results).lower().strip()
            # 3. FUZZY FIXES (Cleaning the sonar font typos)
            # Fix common typos: 't' -> '9', '&' -> '8', 'l' -> '1'
            corrections = {
                't': '7',
                '&': '8',
                'l': '1',
                'o': '0',
                's': '5',
                'i': '1',
                'z': '2',
                'b': '6',
                'g': '9',
                'q': '9',
                'd': '0',

                
            }
            
            clean_text = ""
            for char in raw_text:
                clean_text += corrections.get(char, char)
            
            print(f"EasyOCR Raw: '{raw_text}' -> Cleaned: '{clean_text}'")

            # 4. REGEX with Fallback
            # Strategy A: Try to find a standard decimal (0.9)
            depth_match = re.search(r"(\d+[\.\,]\d)", clean_text)
            if depth_match:
                return f"{depth_match.group(1).replace(',', '.')}"
            
            # Strategy B: If dot is missing but we have 2 numbers (like '09m')
            digits = re.sub("[^0-9]", "", clean_text)
            if len(digits) >= 2 and digits.startswith('0'):
                return f"0.{digits[1]}"
            elif len(digits) >= 2:
                # Handle 1.2, 1.5, etc.
                return f"{digits[0]}.{digits[1]}"

            return "Unknown"
        except Exception:
            return "Unknown"
          
    def run_inference(self, frame):
        depth_val = self._extract_depth(frame)
        # print(f"Depth Meter: {depth_val}m")

        """Run inference based on detection type"""
        if self.current_detection_type == "Fish Detection":
            frame, detected, anomalies, fish_count = self._run_fish_detection(frame)
            return frame, detected, anomalies, fish_count, depth_val
        else:
            frame, detected, anomalies = self._run_yolo_detection(frame)
            return frame, detected, anomalies, depth_val
    
    def _run_yolo_detection(self, frame):
        
        """Run YOLO inference for Asset Detection"""
        if not self.model:
            return frame, [], []
            
        try:
            img_gray = frame.convert("L")
            img_resized = img_gray.resize((640, 640))
            img_input = np.repeat(np.array(img_resized)[:, :, np.newaxis], 3, axis=2)

            results = self.model.predict(img_input, conf=0.25, verbose=False)[0]
        except Exception as e:
            print(f"Inference error: {e}")
            return frame, [], []

        draw = ImageDraw.Draw(frame)
        detected_texts = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_scale = frame.width / 640
                y_scale = frame.height / 640
                x1, x2 = x1 * x_scale, x2 * x_scale
                y1, y2 = y1 * y_scale, y2 * y_scale

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{self.model.names[cls_id]} {conf:.2f}"
                detected_texts.append(label)

                # Draw box
                draw.rectangle([x1, y1, x2, y2], outline="green", width=10)
                font = ImageFont.load_default()
                draw.text((x1, y1 - 10), label, fill="yellow", font=font)

        # Check for anomalies
        anomalies = self.anomaly_detector.analyze_detections(detected_texts, self.current_detection_type)
        
        return frame, detected_texts, anomalies
    
    def _run_fish_detection(self, frame):
        """Run fish detection using regression model"""
        if not self.fish_model:
            return frame, ["Fish model not loaded"], []
        
        try:
            # Use same crop logic as training model: left 75% and top 55%
            h_frame, w_frame = frame.height, frame.width
            x_cutoff = int(w_frame * 0.75)  # Left 75% 
            y_cutoff = int(h_frame * 0.55)  # Top 55%
            roi_frame = frame.crop((0, 450, x_cutoff, y_cutoff))
            
            # Get signal score
            score = self._get_fish_signal_score(roi_frame)
            
            # Predict fish count
            predicted_count = max(0, int(self.fish_model.predict(np.array([[score]]))[0]))
            
            # Check for anomalies
            anomalies = self.anomaly_detector.analyze_fish_count(predicted_count, score)
            
            # Draw visualization
            draw = ImageDraw.Draw(frame)
            font = ImageFont.load_default()
            
            # Draw ROI boundary (match actual crop region)
            draw.rectangle([0, 450, x_cutoff, y_cutoff], outline="cyan", width=2)
            
            # Draw fish count info
            text = f"Fish Count: {predicted_count}"
            score_text = f"Signal Score: {int(score)}"
            # print(text + " | " + score_text)
            
            draw.rectangle([10, 10, 250, 50], fill="black")
            draw.text((15, 15), text, fill="cyan", font=font)
            draw.text((15, 30), score_text, fill="cyan", font=font)
            
            detected_texts = [f"Fish Count: {predicted_count}", f"Score: {int(score)}"]
            
            return frame, detected_texts, anomalies, predicted_count
            
        except Exception as e:
            print(f"Fish detection error: {e}")
            return frame, [], []
        
    def get_phone_location(self):
        """Retrieve the phone's GPS coordinates via ADB"""
        if self.running_source != "phone":
            return None, "Location only available for phone connection."

        try:
            import os
            # Command to dump location information from Android
            # We filter for 'last known' or 'fused' location data
            adb_cmd = [scrcpy_path.replace("scrcpy.exe", "adb.exe"), "shell", "dumpsys location | grep -m 1 'last location'"]
            
            # Execute command
            result = subprocess.check_output(adb_cmd, stderr=subprocess.STDOUT).decode('utf-8')
            
            # Use Regex to find latitude and longitude in the dump output
            # Output typically looks like: "last location=Location[fused 3.123456,101.654321 ...]"
            coords = re.findall(r"([-+]?\d+\.\d+),([-+]?\d+\.\d+)", result)
            
            if coords:
                lat, lon = coords[0]
                # print(f"Phone GPS Coordinates: Lat {lat}, Lon {lon}")
                return (lat, lon), f"Lat: {lat}, Lon: {lon}"
            else:
                return None, "No GPS Cache."
                
        except Exception as e:
            return None, f"ADB Error"
    
    def connect_phone(self):
        """Connect to phone via scrcpy"""
        # Clean up any existing connection first
        self._cleanup_scrcpy()
        
        try:
            self.scrcpy_process = subprocess.Popen(
                [scrcpy_path, "--window-title=" + scrcpy_title, "--no-audio", "--window-x", "-2000"], 
                #  "--max-size=720", "--bit-rate=2M"],  # Optimize for performance
                stdout=subprocess.DEVNULL
                # stderr=subprocess.DEVNULL
            )
            
            # Wait for window with timeout
            for i in range(35):  # Reduced timeout
                time.sleep(0.3)
                
                # Check if process is still running
                if self.scrcpy_process.poll() is not None:
                    self.scrcpy_process = None
                    return False, "Scrcpy process failed to start. Check phone connection."
                
                wins = gw.getWindowsWithTitle(scrcpy_title)
                if wins:
                    self.hwnd = wins[0]._hWnd
                    self.running_source = "phone"
                    # --- NEW: Get location on connection ---
                    coords, loc_msg = self.get_phone_location()
                    return True, f"Phone connected. {loc_msg}"
                    # return True, "Phone connected successfully."
            
            # Timeout - cleanup and return error
            self._cleanup_scrcpy()
            return False, "Connection timeout. Make sure phone is connected and USB debugging is enabled."
            
        except Exception as e:
            self._cleanup_scrcpy()
            return False, f"Failed to start scrcpy: {str(e)}"
    
    def _cleanup_scrcpy(self):
        """Clean up scrcpy process and window"""
        if self.scrcpy_process:
            try:
                self.scrcpy_process.terminate()
                self.scrcpy_process.wait(timeout=2)
            except:
                try:
                    self.scrcpy_process.kill()
                except:
                    pass
            self.scrcpy_process = None
        
        # Close any existing scrcpy windows
        try:
            wins = gw.getWindowsWithTitle(scrcpy_title)
            for win in wins:
                win.close()
        except:
            pass
        
        self.hwnd = None
    
    def load_video(self, file_path):
        """Load video file"""
        try:
            self.video_cap = cv2.VideoCapture(file_path)
            if not self.video_cap.isOpened():
                return False, "Could not open video file."
            
            # Test if we can read the first frame
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.release()
                return False, "Video file appears to be corrupted or empty."
            
            # Reset to beginning
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.running_source = "video"
            return True, f"Video loaded: {file_path.split('/')[-1]}"
        except Exception as e:
            return False, f"Error loading video: {str(e)}"
    
    def get_frame(self):
        """Get current frame based on source"""
        if self.running_source == "phone" and self.hwnd:
            return self.capture_window(self.hwnd)
        elif self.running_source == "video" and self.video_cap:
            ret, frame_bgr = self.video_cap.read()
            if not ret:
                return None
            return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return None
    
    def is_detecting(self):
        """Check if detection is currently running"""
        return self.running_source is not None
    
    def reset_video(self):
        """Reset video source"""
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self._cleanup_scrcpy()
        self.running_source = None
    
    def cleanup(self):
        """Clean up resources"""
        self.reset_video()

    def _get_fish_signal_score(self, roi_image):
        """Get fish signal score from ROI image"""
        try:
            roi_np = cv2.cvtColor(np.array(roi_image), cv2.COLOR_RGB2BGR)
            h_roi, w_roi = roi_np.shape[:2]
            
            if h_roi == 0 or w_roi == 0:
                return 0
            
            hsv = cv2.cvtColor(roi_np, cv2.COLOR_BGR2HSV)
            
            # Use exact same HSV ranges and filtering as training model
            lower_fish_1 = np.array([0, 100, 100])
            upper_fish_1 = np.array([40, 255, 255])
            lower_fish_2 = np.array([170, 100, 100])
            upper_fish_2 = np.array([179, 255, 255])
            
            mask_1 = cv2.inRange(hsv, lower_fish_1, upper_fish_1)
            mask_2 = cv2.inRange(hsv, lower_fish_2, upper_fish_2)
            total_mask = mask_1 | mask_2
            
            # Apply same morphological filtering as training (90% kernel width)
            kernel_width = int(w_roi * 0.9)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
            noise_mask = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
            
            # Top bar filter (same as training)
            top_bar_mask = np.zeros_like(total_mask)
            contours, _ = cv2.findContours(total_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
                if y_c < 50 and w_c > w_roi * 0.8:
                    cv2.drawContours(top_bar_mask, [cnt], -1, (255), -1)
            
            # Combine noise masks and subtract from total (same as training)
            noise_mask = noise_mask | top_bar_mask
            fish_only_mask = cv2.subtract(total_mask, noise_mask)
            
            return cv2.countNonZero(fish_only_mask)
            
        except Exception as e:
            print(f"Signal score error: {e}")
            return 0

if __name__ == "__main__":
    print("--- 📱 STANDALONE PHONE OCR TEST MODE ---")
    
    # 1. Initialize Backend
    backend = DetectionBackend()
    
    # 2. Connect to Phone
    print("Attempting to connect to phone...")
    success, msg = backend.connect_phone()
    print(f"Connection Status: {msg}")
    
    if success:
        print("✅ Phone connected! Starting live OCR stream...")
        print("Press 'q' in the debug window to exit.")
        time.sleep(1) # Let stream stabilize
        
        while True:
            # 3. Capture Frame
            frame = backend.get_frame()
            
            if frame:
                w, h = frame.size
                
                # --- ADJUST COORDINATES HERE TO TEST ---
                # Phone Status Bar is usually ~50-80px tall. 
                # Try starting 'top' at 0 or 40.
                left = 0
                top = 60       # Try changing this if you see the battery icon
                width = int(w * 0.30)  # 40% width to catch the text
                height = 53  # Tall box to ensure we don't miss it
                # # ---------------------------------------
                
                right = left + width
                bottom = top + height
                # left, top, right, bottom = 0, 100, int(w * 0.25), int(h * 0.11)
                # Crop
                roi_depth = frame.crop((left, top, right, bottom))
                
                # Convert to OpenCV for display
                roi_cv = cv2.cvtColor(np.array(roi_depth), cv2.COLOR_RGB2BGR)
                
                # 4. SHOW THE WINDOW (This will pop out now!)
                cv2.imshow("Live Phone ROI - Press 'q' to quit", roi_cv)
                
                # 5. Run OCR (Same logic as your function)
                roi_gray = cv2.cvtColor(roi_cv, cv2.COLOR_BGR2GRAY)
                results = backend.reader.readtext(roi_gray, detail=0)
                
                # Print result to console
                if results:
                    text = " ".join(results)
                    print(f"👀 AI Sees: '{text}'")
                else:
                    print(".", end="", flush=True) # Print dots if nothing found
                
                # Exit loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Waiting for frame...")
                time.sleep(0.1)
                
        # Cleanup
        backend.cleanup()
        cv2.destroyAllWindows()
        print("\nTest complete.")
    else:
        print("❌ Could not connect to phone. Check USB debugging.")