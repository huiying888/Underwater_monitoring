# 🐠 Underwater Asset & Fish Detection System

A real-time computer vision application designed for underwater monitoring. This system integrates **YOLOv11** for asset detection, a **Random Forest Regression model** for fish estimation, and **EasyOCR** for reading digital depth meters from sonar feeds.

It supports both **Video File analysis** and **Live Phone Streaming** via USB (using scrcpy).

## 🚀 Features
* **Dual Detection Modes:** Switch between Asset Detection (YOLO) and Fish Counting (Regression).
* **Live Telemetry:** Automated reading of Sonar Depth (OCR) and Phone GPS Geolocation.
* **Mobile Integration:** Low-latency screen mirroring from Android devices via USB.
* **Anomaly Detection:** Real-time alerts for critical shifts, moving structures, or unexpected fish presence.
* **Reporting:** Automatic logging to CSV and session export capabilities.

---

## 🛠️ Prerequisites

Before installing, ensure you have:
1.  **Python 3.9+** installed.
2.  **Android Phone** with Developer Options enabled.
3.  **USB Cable** for data transfer.

---

## 📥 Installation Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Underwater-Detection.git](https://github.com/YOUR_USERNAME/Underwater-Detection.git)
cd Underwater-Detection
