import time
import numpy as np
from datetime import datetime
import winsound
import threading
from data_logger import DataLogger
from email_alerts import EmailAlerter


class AnomalyDetector:
    def __init__(self):
        # Define expected vs anomaly classes for each detection type
        self.asset_expected = {'empty_water_tank', 'near_standing_structure', 'near_moving_standing_structure', 'rope', 'lying_oil_rig', 'lying_pipe', 'protrude_pipe'}
        self.asset_anomalies = {
            'major_shift': 'CRITICAL',
            'moving_camera_stick': 'MEDIUM',
            'fish': 'HIGH',  # Fish shouldn't be in asset monitoring
            'fishing_pond': 'HIGH'  # Fishing pond shouldn't be in asset monitoring
        }
        
        self.fish_expected = {'fish', 'fishing_pond'}
        self.fish_anomalies = {
            'lying_oil_rig': 'CRITICAL',  # Asset shouldn't be in fish monitoring
            'lying_pipe': 'CRITICAL',
            'empty_water_tank': 'HIGH',
            'major_shift': 'HIGH',
            'moving_camera_stick': 'MEDIUM',
            'near_moving_standing_structure': 'MEDIUM',
            'near_standing_structure': 'MEDIUM',
            'protrude_pipe': 'HIGH',
            'rope': 'LOW'
        }
        
        # Alert thresholds
        self.alert_cooldown = 5  # seconds between same alerts
        self.last_alerts = {}
        self.detection_history = []
        self.active_anomalies = {}  # Store active anomalies with timestamps
        self.anomaly_display_duration = 5  # seconds to keep showing anomaly
        
        # Initialize logging and email alerts
        self.data_logger = DataLogger()
        self.email_alerter = EmailAlerter()
        
        # Fish count monitoring
        self.fish_count_history = []
        self.fish_drop_start_time = None
        self.fish_drop_threshold = 0.2  # 20% drop (demo: more sensitive)
        self.fish_drop_duration = 3  # 3 seconds (demo: quick confirmation)
        self.low_count_start_time = None  # Track when count goes below 50
        self.drop_baseline = None  # Store original baseline for recovery check
        
    def analyze_detections(self, detections, detection_type):
        """Analyze detections for anomalies"""
        anomalies = []
        current_time = time.time()
        
        for detection in detections:
            class_name = detection.split(' ')[0]  # Extract class name from "class_name confidence"
            
            # Check if this is an anomaly
            severity = self._get_anomaly_severity(class_name, detection_type)
            if severity:
                anomaly_key = f"{detection_type}_{class_name}"
                
                # Check cooldown
                if self._should_alert(anomaly_key, current_time):
                    anomaly = {
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'type': detection_type,
                        'class': class_name,
                        'severity': severity,
                        'detection': detection
                    }
                    anomalies.append(anomaly)
                    self.last_alerts[anomaly_key] = current_time
                    # Store as active anomaly
                    self.active_anomalies[anomaly_key] = {
                        'anomaly': anomaly,
                        'last_seen': current_time
                    }
                    
                    # Log anomaly and send email alert
                    self.data_logger.log_anomaly(anomaly)
                    self.email_alerter.send_anomaly_alert(anomaly)
            else:
                # Update last seen time for existing anomalies
                anomaly_key = f"{detection_type}_{class_name}"
                if anomaly_key in self.active_anomalies:
                    self.active_anomalies[anomaly_key]['last_seen'] = current_time
        
        # Store detection history and log data
        if detections:
            self.detection_history.append({
                'timestamp': current_time,
                'type': detection_type,
                'count': len(detections),
                'detections': detections
            })
            
            # Log detections to CSV
            self.data_logger.log_detection(detection_type, detections, anomalies)
            
            # Keep only last 100 entries
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)
        
        # Clean up old anomalies
        self._cleanup_old_anomalies(current_time)
        
        return anomalies
    
    def get_active_anomalies(self):
        """Get currently active anomalies for display"""
        current_time = time.time()
        active = []
        
        for key, data in self.active_anomalies.items():
            if (current_time - data['last_seen']) <= self.anomaly_display_duration:
                active.append(data['anomaly'])
        
        return active
    
    def _cleanup_old_anomalies(self, current_time):
        """Remove old anomalies that are no longer active"""
        expired_keys = []
        for key, data in self.active_anomalies.items():
            if (current_time - data['last_seen']) > self.anomaly_display_duration:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.active_anomalies[key]
    
    def _get_anomaly_severity(self, class_name, detection_type):
        """Get severity level for detected class"""
        class_lower = class_name.lower()
        
        if detection_type == "Asset Detection":
            # In asset monitoring, only certain things are anomalies
            return self.asset_anomalies.get(class_lower)
        elif detection_type == "Fish Detection":
            # In fish monitoring, non-fish/pond things are anomalies
            return self.fish_anomalies.get(class_lower)
        return None
    
    def _should_alert(self, anomaly_key, current_time):
        """Check if we should alert for this anomaly"""
        last_alert = self.last_alerts.get(anomaly_key, 0)
        return (current_time - last_alert) > self.alert_cooldown
    
    def trigger_alert(self, anomaly):
        """Trigger alert for anomaly"""
        # Sound alert based on severity
        if anomaly['severity'] == 'CRITICAL':
            threading.Thread(target=self._play_critical_alert, daemon=True).start()
        elif anomaly['severity'] == 'HIGH':
            threading.Thread(target=self._play_high_alert, daemon=True).start()
        elif anomaly['severity'] == 'MEDIUM':
            threading.Thread(target=self._play_medium_alert, daemon=True).start()
    
    def _play_critical_alert(self):
        """Play critical alert sound"""
        for _ in range(3):
            winsound.Beep(1000, 300)
            time.sleep(0.1)
    
    def _play_high_alert(self):
        """Play high priority alert sound"""
        for _ in range(2):
            winsound.Beep(800, 200)
            time.sleep(0.1)
    
    def _play_medium_alert(self):
        """Play medium priority alert sound"""
        winsound.Beep(600, 150)
    
    def get_alert_summary(self):
        """Get summary of recent alerts"""
        recent_time = time.time() - 300  # Last 5 minutes
        recent_detections = [d for d in self.detection_history if d['timestamp'] > recent_time]
        
        summary = {
            'total_detections': len(recent_detections),
            'asset_detections': len([d for d in recent_detections if d['type'] == 'Asset Detection']),
            'fish_detections': len([d for d in recent_detections if d['type'] == 'Fish Detection']),
            'last_detection': recent_detections[-1]['timestamp'] if recent_detections else None
        }
        
        return summary
    
    def export_session_data(self):
        """Export current session data"""
        return self.data_logger.export_session_summary(self.detection_history)
    
    def configure_email_alerts(self, smtp_server, smtp_port, sender_email, sender_password, recipient_emails):
        """Configure email alert settings"""
        self.email_alerter.configure(smtp_server, smtp_port, sender_email, sender_password, recipient_emails)
    
    def analyze_fish_count(self, fish_count, signal_score):
        """Analyze fish count for anomalies (drops)"""
        current_time = time.time()
        anomalies = []
        
        # Store fish count history
        self.fish_count_history.append({
            'timestamp': current_time,
            'count': fish_count,
            'score': signal_score
        })
        
        # Keep only last 50 entries
        if len(self.fish_count_history) > 50:
            self.fish_count_history.pop(0)
        
        # Need at least 10 frames for comparison
        if len(self.fish_count_history) >= 10:
            recent_avg = sum(h['count'] for h in self.fish_count_history[-10:]) / 10
            baseline_avg = sum(h['count'] for h in self.fish_count_history[-30:-10]) / 20 if len(self.fish_count_history) >= 30 else recent_avg
            
            # Detect significant drop
            if baseline_avg > 20 and recent_avg < baseline_avg * (1 - self.fish_drop_threshold):
                if self.fish_drop_start_time is None:
                    self.fish_drop_start_time = current_time
                    self.drop_baseline = baseline_avg  # Store original baseline
                    print(f"Fish drop detected: {baseline_avg:.1f} -> {recent_avg:.1f}")
                elif (current_time - self.fish_drop_start_time) >= 3:
                    # Check if recovery is insufficient after 3 seconds
                    recovery_threshold = self.drop_baseline * 0.9  # Need 90% recovery
                    if recent_avg < recovery_threshold:
                        # CRITICAL: Drop detected but insufficient recovery
                        anomaly_key = "fish_drop_critical"
                        if self._should_alert(anomaly_key, current_time):
                            anomaly = {
                                'timestamp': datetime.now().strftime('%H:%M:%S'),
                                'type': 'Fish Detection',
                                'class': 'fish_drop_critical',
                                'severity': 'CRITICAL',
                                'detection': f'Fish count dropped from {int(self.drop_baseline)} to {int(recent_avg)} - No recovery!'
                            }
                            anomalies.append(anomaly)
                            self.last_alerts[anomaly_key] = current_time
                            self.active_anomalies[anomaly_key] = {
                                'anomaly': anomaly,
                                'last_seen': current_time
                            }
                            print(f"CRITICAL FISH DROP ALERT: {self.drop_baseline:.1f} -> {recent_avg:.1f} (Need {recovery_threshold:.1f})")
                            
                            # Log and email
                            self.data_logger.log_anomaly(anomaly)
                            self.email_alerter.send_fish_drop_alert(anomaly, self.drop_baseline, recent_avg)
                    else:
                        # Regular HIGH alert for sustained drop
                        anomaly_key = "fish_count_drop"
                        if self._should_alert(anomaly_key, current_time):
                            anomaly = {
                                'timestamp': datetime.now().strftime('%H:%M:%S'),
                                'type': 'Fish Detection',
                                'class': 'fish_count_drop',
                                'severity': 'HIGH',
                                'detection': f'Fish count dropped from {int(self.drop_baseline)} to {int(recent_avg)}'
                            }
                            anomalies.append(anomaly)
                            self.last_alerts[anomaly_key] = current_time
                            self.active_anomalies[anomaly_key] = {
                                'anomaly': anomaly,
                                'last_seen': current_time
                            }
                            print(f"FISH DROP ALERT: {self.drop_baseline:.1f} -> {recent_avg:.1f}")
                            
                            # Log and email
                            self.data_logger.log_anomaly(anomaly)
                            self.email_alerter.send_fish_drop_alert(anomaly, self.drop_baseline, recent_avg)
            # Recovery requires getting back to 95% of original baseline
            elif self.drop_baseline and recent_avg >= self.drop_baseline * 0.95:
                if self.fish_drop_start_time is not None:
                    print(f"Fish count fully recovered: {recent_avg:.1f}")
                self.fish_drop_start_time = None
                self.drop_baseline = None
        
        # Check for low fish count (below 60 for 3+ seconds, recover at 80)
        if fish_count < 80:
            if self.low_count_start_time is None:
                self.low_count_start_time = current_time
                print(f"Low fish count detected: {fish_count}")
            elif (current_time - self.low_count_start_time) >= 3:
                anomaly_key = "fish_count_low"
                if self._should_alert(anomaly_key, current_time):
                    anomaly = {
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'type': 'Fish Detection',
                        'class': 'fish_count_low',
                        'severity': 'MEDIUM',
                        'detection': f'Fish count critically low: {fish_count}'
                    }
                    anomalies.append(anomaly)
                    self.last_alerts[anomaly_key] = current_time
                    self.active_anomalies[anomaly_key] = {
                        'anomaly': anomaly,
                        'last_seen': current_time
                    }
                    print(f"LOW FISH COUNT ALERT: {fish_count}")
                    
                    # Log and email
                    self.data_logger.log_anomaly(anomaly)
                    self.email_alerter.send_anomaly_alert(anomaly)
        elif fish_count >= 80:
            if self.low_count_start_time is not None:
                print(f"Fish count recovered above 80: {fish_count}")
            self.low_count_start_time = None
        
        # Log fish count data
        self.data_logger.log_fish_count(fish_count, signal_score)
        
        return anomalies

    
    def test_email_configuration(self):
        """Test email configuration"""
        return self.email_alerter.send_test_email()
    def test_email_configuration(self):
        """Test email configuration"""
        return self.email_alerter.send_test_email()