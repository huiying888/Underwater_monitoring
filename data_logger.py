import csv
import os
from datetime import datetime
import json


class DataLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.ensure_log_directory()
        
    def ensure_log_directory(self):
        """Create logs directory if it doesn't exist"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def log_detection(self, detection_type, detections, anomalies=None):
        """Log detection data to CSV"""
        timestamp = datetime.now()
        date_str = timestamp.strftime('%Y-%m-%d')
        
        # Detection log file
        detection_file = os.path.join(self.log_dir, f"detections_{date_str}.csv")
        
        # Check if file exists to write header
        file_exists = os.path.exists(detection_file)
        
        with open(detection_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if new file
            if not file_exists:
                writer.writerow(['Timestamp', 'Detection_Type', 'Class', 'Confidence', 'Is_Anomaly', 'Severity'])
            
            # Write detections
            for detection in detections:
                parts = detection.split(' ')
                class_name = parts[0]
                confidence = parts[1] if len(parts) > 1 else 'N/A'
                
                # Check if this detection is an anomaly
                is_anomaly = False
                severity = 'NORMAL'
                if anomalies:
                    for anomaly in anomalies:
                        if anomaly['class'] == class_name:
                            is_anomaly = True
                            severity = anomaly['severity']
                            break
                
                writer.writerow([
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    detection_type,
                    class_name,
                    confidence,
                    is_anomaly,
                    severity
                ])
    
    def log_anomaly(self, anomaly):
        """Log anomaly to separate file"""
        timestamp = datetime.now()
        date_str = timestamp.strftime('%Y-%m-%d')
        
        anomaly_file = os.path.join(self.log_dir, f"anomalies_{date_str}.csv")
        file_exists = os.path.exists(anomaly_file)
        
        with open(anomaly_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(['Timestamp', 'Detection_Type', 'Class', 'Severity', 'Full_Detection'])
            
            writer.writerow([
                timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                anomaly['type'],
                anomaly['class'],
                anomaly['severity'],
                anomaly['detection']
            ])
    
    def export_session_summary(self, detection_history):
        """Export session summary to JSON"""
        timestamp = datetime.now()
        filename = f"session_summary_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        summary = {
            'session_start': timestamp.isoformat(),
            'total_detections': len(detection_history),
            'detection_counts': {},
            'detections': []
        }
        
        for entry in detection_history:
            detection_time = datetime.fromtimestamp(entry['timestamp'])
            summary['detections'].append({
                'timestamp': detection_time.isoformat(),
                'type': entry['type'],
                'count': entry['count'],
                'detections': entry['detections']
            })
            
            # Count by type
            det_type = entry['type']
            summary['detection_counts'][det_type] = summary['detection_counts'].get(det_type, 0) + entry['count']
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        return filepath
    
    def log_fish_count(self, fish_count, signal_score):
        """Log fish count data to CSV"""
        timestamp = datetime.now()
        date_str = timestamp.strftime('%Y-%m-%d')
        
        fish_file = os.path.join(self.log_dir, f"fish_counts_{date_str}.csv")
        file_exists = os.path.exists(fish_file)
        
        with open(fish_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(['Timestamp', 'Fish_Count', 'Signal_Score'])
            
            writer.writerow([
                timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                fish_count,
                int(signal_score)
            ])