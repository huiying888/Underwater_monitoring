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
3. Important: Rename the extracted folder to scrcpy-win64-v3.3.1 (or update config.py to match).

### 4. Setup Your Android Phone
To use the live feed feature, you must enable USB Debugging:
1. Go to Settings > About Phone.
2. Tap Build Number 7 times until it says "You are a developer".
3. Go back to Settings > System > Developer Options.
4. Enable USB Debugging.
5. Connect your phone to the PC via USB.
6. On your phone screen, a popup will appear asking to "Allow USB Debugging?". Check "Always allow" and tap Allow.


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






