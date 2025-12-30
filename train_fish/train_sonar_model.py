# train_sonar_model.py
import cv2
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import os

# --- 1. SIGNAL SCORE FUNCTION ---
def get_signal_score(roi_image):
    """
    Analyzes the 'Graph ROI' (top 75% of the frame).
    It uses morphological operations to surgically remove the
    top bar and the bottom ground reflection, even if fish
    are touching them.
    """
    if not isinstance(roi_image, np.ndarray):
        roi_image = cv2.cvtColor(np.array(roi_image), cv2.COLOR_RGB2BGR)

    h_roi, w_roi = roi_image.shape[:2]
    if h_roi == 0 or w_roi == 0:
        return 0

    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

    # ==========================================================
    # PASTE YOUR 4 TUNED LINES FROM tune_mask_fish.py
    lower_fish_1 = np.array([0, 100, 100])
    upper_fish_1 = np.array([40, 255, 255])
    lower_fish_2 = np.array([170, 100, 100])
    upper_fish_2 = np.array([179, 255, 255])
    # ==========================================================

    mask_1 = cv2.inRange(hsv, lower_fish_1, upper_fish_1)
    mask_2 = cv2.inRange(hsv, lower_fish_2, upper_fish_2)
    # This is the "Total Mask" (Fish + Ground + Top Bar)
    total_mask = mask_1 | mask_2

    # --- This is our new, smarter filter ---

    # 1. Create a "ground filter"
    # A long, horizontal kernel will find *only* wide, flat objects.
    # We use a kernel that is 50% of the ROI width.
    kernel_width = int(w_roi * 0.9)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    
    # "Opening" removes small objects (like fish) and leaves
    # only the long, flat objects (ground and top bar).
    noise_mask = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # 2. Create a "top bar filter" (just in case)
    # We find anything wide in the top 50 pixels
    top_bar_mask = np.zeros_like(total_mask)
    contours, _ = cv2.findContours(total_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        if y_c < 50 and w_c > w_roi * 0.8:
            cv2.drawContours(top_bar_mask, [cnt], -1, (255), -1)

    # 3. Combine the noise masks
    noise_mask = noise_mask | top_bar_mask

    # 4. Subtract the noise from the total mask
    # This leaves ONLY the fish clusters.
    fish_only_mask = cv2.subtract(total_mask, noise_mask)
    
    # 5. Get the score from the clean mask
    total_fish_score = cv2.countNonZero(fish_only_mask)

    # # --- YOUR NEW 8000 THRESHOLD ---
    # if total_fish_score < 8000:
    #     return 0
    # # --- END NEW LOGIC ---
        
    return total_fish_score

# --- 2. VIDEO PROCESSING FUNCTION (WITH YOUR NEW CROP) ---
def process_video_for_score(video_path, known_count):
    """
    Processes a full video using your custom crop logic.
    **Includes outlier filtering for videos with fish.**
    """
    cap = cv2.VideoCapture(video_path)
    scores = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # --- THIS IS YOUR NEW CROP LOGIC ---
        h_frame, w_frame = frame.shape[:2]
        x_cutoff = int(w_frame * 0.75) # Get left 75%
        y_cutoff = int(h_frame * 0.55) # Get top 55%
        
        # Make sure crop is valid
        if 100 >= y_cutoff or 0 >= x_cutoff:
            continue
            
        top_roi = frame[450:y_cutoff, 0:x_cutoff]
        # ---
        
        if top_roi.size == 0:
            continue
            
        score = get_signal_score(top_roi)
        scores.append(score)
        
    cap.release()
    
    if not scores: 
        return 0

    if known_count == 0:
        return np.mean(scores)
    
    avg_score = np.mean(scores)
    std_dev = np.std(scores)
    threshold = avg_score - (3 * std_dev)
    threshold = max(1, threshold) 
    
    cleaned_scores = [s for s in scores if s > threshold]
    
    if not cleaned_scores:
        return avg_score
        
    final_avg = np.mean(cleaned_scores)
    print(f"  -> Data cleaning: Original avg: {avg_score:.0f}, Cleaned avg: {final_avg:.0f} (removed {len(scores) - len(cleaned_scores)} outliers)")
    return final_avg

# --- 3. DATASET DEFINITION & TRAINING ---
def train_model():
    print("Processing videos to create dataset...")

    # ==========================================================
    # UPDATE THIS DICTIONARY WITH YOUR VIDEO FILES
    video_files = {
        r"C:\Users\HP\Documents\FYP DATASET\fish_150(1).mov": 150,
        r"C:\Users\HP\Documents\FYP DATASET\fish_300.MOV": 300,
        r"C:\Users\HP\Documents\FYP DATASET\fishingpond.MOV": 0
    }
    # ==========================================================

    X_train_scores = []
    y_train_counts = []

    for path, count in video_files.items():
        if not os.path.exists(path):
            print(f"Warning: Video file not found: {path}. Skipping.")
            continue
            
        print(f"Processing {path} (Known Count: {count})...")
        avg_score = process_video_for_score(path, count)
        
        X_train_scores.append(avg_score)
        y_train_counts.append(count)
        print(f"  -> Using Average Score: {avg_score:.0f}")

    if len(X_train_scores) < 2:
        print("\nError: Not enough data. Need at least 2 valid videos.")
        return

    X = np.array(X_train_scores).reshape(-1, 1)
    y = np.array(y_train_counts)

    # --- 4. TRAIN AND SAVE MODEL ---
    print("\nTraining regression model...")
    model = LinearRegression()
    # model = RandomForestRegressor(n_estimators=10, random_state=42)
    # model = Pipeline([
    #     ('poly', PolynomialFeatures(degree=2)),
    #     ('linear', LinearRegression())
    # ])
    model.fit(X, y)

    model_filename = 'sonar_model_linearregression.joblib'
    joblib.dump(model, model_filename)

    print(f"\nModel training complete!")
    print(f"Model saved as: {model_filename}")

    print("\n--- Model Test ---")
    for i in range(len(X)):
        score = X[i].reshape(1, -1)
        prediction = model.predict(score)
        print(f"  Score: {X[i][0]:.0f}, Actual: {y[i]}, Predicted: {prediction[0]:.0f}")

# --- 5. RUN TRAINING ---
if __name__ == "__main__":
    train_model()