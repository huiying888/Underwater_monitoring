# tune_mask_fish.py
import cv2
import numpy as np

# --- SETTINGS ---
# 1. Put one of your cropped sonar images here (like image_6f2599.png)
IMAGE_PATH = r'train_fish\fish2.JPG' 
# ---

def nothing(x):
    pass

# Load the image
roi = cv2.imread(IMAGE_PATH)
if roi is None:
    print(f"Error: Could not load image from {IMAGE_PATH}")
    exit()
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Create a window
cv2.namedWindow('Tune Mask - Find the FISH cluster', cv2.WINDOW_NORMAL)

# Red/Orange/Yellow are usually in the low H range (0-40)
# and the "wrap-around" H range (170-180)
cv2.createTrackbar('H_min_1', 'Tune Mask - Find the FISH cluster', 0, 179, nothing)
cv2.createTrackbar('S_min', 'Tune Mask - Find the FISH cluster', 100, 255, nothing)
cv2.createTrackbar('V_min', 'Tune Mask - Find the FISH cluster', 100, 255, nothing)
cv2.createTrackbar('H_max_1', 'Tune Mask - Find the FISH cluster', 40, 179, nothing)
cv2.createTrackbar('S_max', 'Tune Mask - Find the FISH cluster', 255, 255, nothing)
cv2.createTrackbar('V_max', 'Tune Mask - Find the FISH cluster', 255, 255, nothing)

# Add a second range for "wrap-around" red
cv2.createTrackbar('H_min_2', 'Tune Mask - Find the FISH cluster', 170, 179, nothing)
cv2.createTrackbar('H_max_2', 'Tune Mask - Find the FISH cluster', 179, 179, nothing)

print("Adjust sliders until 'Fish Mask' shows ONLY the red/orange/yellow cluster.")
print("The background (blue, white, purple) should be BLACK.")
print("Press 'q' to quit and print the final values.")

while True:
    h_min_1 = cv2.getTrackbarPos('H_min_1', 'Tune Mask - Find the FISH cluster')
    s_min = cv2.getTrackbarPos('S_min', 'Tune Mask - Find the FISH cluster')
    v_min = cv2.getTrackbarPos('V_min', 'Tune Mask - Find the FISH cluster')
    h_max_1 = cv2.getTrackbarPos('H_max_1', 'Tune Mask - Find the FISH cluster')
    s_max = cv2.getTrackbarPos('S_max', 'Tune Mask - Find the FISH cluster')
    v_max = cv2.getTrackbarPos('V_max', 'Tune Mask - Find the FISH cluster')
    h_min_2 = cv2.getTrackbarPos('H_min_2', 'Tune Mask - Find the FISH cluster')
    h_max_2 = cv2.getTrackbarPos('H_max_2', 'Tune Mask - Find the FISH cluster')

    lower_1 = np.array([h_min_1, s_min, v_min])
    upper_1 = np.array([h_max_1, s_max, v_max])
    mask_1 = cv2.inRange(hsv, lower_1, upper_1)

    lower_2 = np.array([h_min_2, s_min, v_min])
    upper_2 = np.array([h_max_2, s_max, v_max])
    mask_2 = cv2.inRange(hsv, lower_2, upper_2)

    fish_mask = mask_1 | mask_2

    cv2.imshow('Original ROI', roi)
    cv2.imshow('Fish Mask (Result)', fish_mask)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()

print("\n--- Your Tuned Values ---")
print(f"lower_fish_1 = np.array([{h_min_1}, {s_min}, {v_min}])")
print(f"upper_fish_1 = np.array([{h_max_1}, {s_max}, {v_max}])")
print(f"lower_fish_2 = np.array([{h_min_2}, {s_min}, {v_min}])")
print(f"upper_fish_2 = np.array([{h_max_2}, {s_max}, {v_max}])")
print("\nCopy these 4 lines into Step 4.")