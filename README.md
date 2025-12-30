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


