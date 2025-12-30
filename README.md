# 🐠 Underwater Asset & Fish Detection System

A real-time computer vision application designed for underwater monitoring. This system integrates **YOLOv11** for asset detection, a **Random Forest Regression model** for fish estimation, and **EasyOCR** for reading digital depth meters from sonar feeds.

It supports both **Video File analysis** and **Live Phone Streaming** via USB (using scrcpy).

**⚠️ Hardware Note:** This system is specifically calibrated for the **Erchang F68 Fish Finder** interface.


## 🚀 Features
* **Dual Detection Modes:** Switch between Asset Detection (YOLO) and Fish Counting (Regression).
* **Live Telemetry:** Automated reading of Sonar Depth (OCR) and Phone GPS Geolocation.
* **Mobile Integration:** Low-latency screen mirroring from Android devices via USB.
* **Anomaly Detection:** Real-time alerts for critical shifts, moving structures, or unexpected fish presence.
* **Reporting:** Automatic logging to CSV and session export capabilities.

---

## 🛠️ Prerequisites & Hardware

To use this system effectively, you need:

### 1. Hardware
* **Erchang F68 Fish Finder** (Sonar Device).
* **Android Phone** (to run the Erchang companion app).
* **USB Cable** (to connect the phone to the PC).
* **PC/Laptop** with Python 3.9+ installed.

### 2. Software Dependencies
* **Python 3.9+**
* **Erchang Fish Finder App** installed on your Android phone.
* **USB Debugging** enabled on the Android phone.

---

## 📥 Installation Guide

### 1. Clone the Repository
```bash
git clone [[https://github.com/huiying888/Underwater_Monitoring.git](https://github.com/huiying888/Underwater_monitoring.git)]
cd Underwater_Monitoring
```

### 2. Install Python Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Setup Scrcpy (For Live Phone Feed)
1. Download the latest Scrcpy release for Windows from [[here](https://github.com/Genymobile/scrcpy/releases)].
2. Extract the downloaded zip file into C Drive
3. Important: Rename the extracted folder to scrcpy-win64-v3.3.1 (or update `config.py` to match).

### 4. Setup Your Android Phone
To use the live feed feature, you must enable USB Debugging:
1. Go to Settings > About Phone.
2. Tap Build Number 7 times until it says "You are a developer".
3. Go back to Settings > System > Developer Options.
4. Enable USB Debugging.
5. Connect your phone to the PC via USB.
6. On your phone screen, a popup will appear asking to "Allow USB Debugging?". Check "Always allow" and tap Allow.
7. Watch this [[video](https://youtu.be/2C2hZT3bpBo?si=ReXDt0Cks5vvjJkb)] for clarification.

---
## ⚙️ Configuration & ROI Calibration

Important: The Optical Character Recognition (OCR) is tuned for a specific screen resolution. If the "Depth" reading says Unknown or returns wrong numbers, you likely need to adjust the Region of Interest (ROI) coordinates.

### 1. Calibrate for Phone (Live Feed)
The system includes a debug mode to let you see exactly what the AI is "seeing".
1. Connect your phone (ensure USB debugging is on).
2. Run the backend test script:
```bash
python detection_backend.py
```
3. A window named "Live Phone ROI" will pop up showing the cropped area.
*If you see the depth numbers: The calibration is correct.
*If you see the battery icon/clock: The box is too high. Increase top.
*If the box is black/empty: The box is too low. Decrease top.

4.To Fix: Open detection_backend.py and adjust the coordinates in the _extract_depth function (approx. line 95):
```bash
else:
    left = 0
    top = 60       # <--- ADJUST THIS VALUE (y-axis start)
    width = int(w * 0.30)
    height = 53    # <--- ADJUST THIS VALUE (box height)
```
### 2. Calibrate for Video Files
If you are analyzing video files and the depth is missing:
1. Open `test_ocr.py`.
2. Edit the bottom line to point to your video file:
```bash
test_depth_view(r'C:\path\to\your\video.mp4')
```
3. Run python `test_ocr.py`.
4. Adjust the left, top, right, bottom variables in test_ocr.py until the white numbers are clearly visible in the popup window.
5. Copy those new values into the `if self.running_source == "video":` section of `detection_backend.py`.
   
---
## 🖥️ How to Use

### 1. Prepare the Sonar
1. Turn on your Erchang F68 Fish Finder.
2. Connect your Android phone to the Fish Finder's WiFi (or Bluetooth) as per the device instructions.
3. Open the Erchang App on your phone and ensure you can see the sonar feed.


### 2. Run the Computer Vision System
Connect your phone to your PC via USB and run:
```bash
python main.py
```

### 3. Start Detection
1. In the GUI, click 📱 Connect Device.
2. The system will mirror your phone screen (showing the Erchang App).
3. Select Detection Type:
* Fish Detection: Counts fish and analyzes signal density.
* Asset Detection: Detects underwater pipes, structures, or debris.
4. Telemetry: The system will automatically read the Depth from the top-left corner of the Erchang app interface and log your GPS location.






