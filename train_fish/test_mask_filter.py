# test_filter_logic.py
import cv2
import numpy as np
import os

# --- 1. SETTINGS TO EDIT ---

# Put the path to your "no fish" video (like fishingpond.MOV or the
# one from image_72e719.png) to test the filter.
VIDEO_PATH = r"C:\Users\HP\Documents\FYP DATASET\fish_150(1).mov"

# ==========================================================
# PASTE YOUR 4 TUNED LINES FROM tune_mask_fish.py
lower_fish_1 = np.array([0, 100, 100])
upper_fish_1 = np.array([40, 255, 255])
lower_fish_2 = np.array([170, 100, 100])
upper_fish_2 = np.array([179, 255, 255])
# ==========================================================


# --- 2. FILTERING FUNCTION (Identical to train_sonar_model.py) ---
def get_signal_score_and_masks(roi_image):
    """
    Analyzes the 'Graph ROI' (top 75% of the frame).
    It finds all red/yellow pixels, then filters out the
    top bar and the bottom ground reflection based on their width and position.
    
    Returns:
        - total_fish_score
        - total_mask (all red/yellow pixels)
        - noise_mask (the parts we are removing)
        - fish_only_mask (the final result)
    """
    if not isinstance(roi_image, np.ndarray):
        roi_image = cv2.cvtColor(np.array(roi_image), cv2.COLOR_RGB2BGR)

    h_roi, w_roi = roi_image.shape[:2]
    if h_roi == 0 or w_roi == 0:
        return 0, None, None, None

    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

    mask_1 = cv2.inRange(hsv, lower_fish_1, upper_fish_1)
    mask_2 = cv2.inRange(hsv, lower_fish_2, upper_fish_2)
    # This is the "Total Mask" (Fish + Ground + Top Bar)
    total_mask = mask_1 | mask_2
    
    kernel = np.ones((3,3), np.uint8)
    total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- This is our filter logic ---
    
    # 1. Create a "ground filter"
    # A long, horizontal kernel will find *only* wide, flat objects.
    kernel_width = int(w_roi * 0.8)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    ground_mask = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # 2. Create a "top bar filter"
    top_bar_mask = np.zeros_like(total_mask)
    contours, _ = cv2.findContours(total_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wide_threshold = w_roi * 0.8
    total_fish_score = 0
    
    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        
        # 1. Filter out the TOP bar (touching top edge AND wide)
        if y_c < 50 and w_c > wide_threshold:
            # Draw this contour onto the top_bar_mask
            cv2.drawContours(top_bar_mask, [cnt], -1, (255), -1)
            continue
            
        # 2. Filter out the BOTTOM ground reflection
        # (touching bottom edge of our ROI AND wide)
        if (y_c + h_c) > (h_roi - 20) and w_c > wide_threshold:
            # We already found this with the ground_mask, so we just continue
            continue
            
        # --- If it's not a wide bar, it's fish! ---
        total_fish_score += cv2.contourArea(cnt)
        
    # Combine the noise masks
    noise_mask = ground_mask | top_bar_mask

    # Subtract the noise from the total mask
    fish_only_mask = cv2.subtract(total_mask, noise_mask)
    
    # Recalculate score from the clean mask for accuracy
    final_score = cv2.countNonZero(fish_only_mask)

    return final_score, total_mask, noise_mask, fish_only_mask

# --- 3. VIDEO PROCESSING & VISUALIZATION ---
def test_video_filter(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    print("\n--- Starting Visual Test ---")
    print("Press 'q' to quit.")
    print("Press 'p' to pause/unpause.")
    
    paused = False
    
    cv2.namedWindow('Original Top 75%', cv2.WINDOW_NORMAL)
    cv2.namedWindow('1. Total Mask (All Red/Yellow)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('2. Noise Filter (Ground/Bar)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('3. FINAL RESULT (Fish Only)', cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video. Looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # --- This is your 3/4 (75%) crop logic ---
            h_frame, w_frame = frame.shape[:2]
            x_cutoff = int(w_frame * 0.75) # Get left 80%
            y_cutoff = int(h_frame * 0.55) # Get top 75%
            top_75_roi = frame[450:y_cutoff, 0:x_cutoff]
            # ---
            
            if top_75_roi.size == 0:
                continue
                
            score, total, noise, fish_only = get_signal_score_and_masks(top_75_roi)
            
            # Print the score to the console
            print(f"Current Fish Score: {score}")

            # Display all the windows
            cv2.imshow('Original Top 75%', top_75_roi)
            cv2.imshow('1. Total Mask (All Red/Yellow)', total)
            cv2.imshow('2. Noise Filter (Ground/Bar)', noise)
            cv2.imshow('3. FINAL RESULT (Fish Only)', fish_only)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

# --- 4. RUN TEST ---
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at: {VIDEO_PATH}")
        print("Please edit the VIDEO_PATH variable in this script.")
    else:
        test_video_filter(VIDEO_PATH)