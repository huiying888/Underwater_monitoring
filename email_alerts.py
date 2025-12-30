import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


class EmailAlerter:
    def __init__(self):
        # Email configuration - update these with your settings
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = "jocelynngieng@gmail.com"  # Set in config
        self.sender_password = "zpmlijmfrtgtpmei"  # Set in config or use app password
        self.recipient_emails = ["jocelynngieng@gmail.com"]  # Set in config
        self.enabled = False
        
    def configure(self, smtp_server, smtp_port, sender_email, sender_password, recipient_emails):
        """Configure email settings"""
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails if isinstance(recipient_emails, list) else [recipient_emails]
        self.enabled = bool(sender_email and sender_password and recipient_emails)
    
    def send_anomaly_alert(self, anomaly):
        """Send email alert for anomaly"""
        if not self.enabled:
            return False
            
        # Only send emails for CRITICAL and HIGH severity
        if anomaly['severity'] not in ['CRITICAL', 'HIGH']:
            return False
        
        # Send email in background thread
        threading.Thread(target=self._send_email_async, args=(anomaly,), daemon=True).start()
        return True
    
    def _send_email_async(self, anomaly):
        """Send email asynchronously"""
        try:
            subject = f"🚨 {anomaly['severity']} UNDERWATER MONITORING ALERT"
            
            body = f"""
UNDERWATER MONITORING SYSTEM ALERT

Alert Details:
- Severity: {anomaly['severity']}
- Detection Type: {anomaly['type']}
- Object Class: {anomaly['class']}
- Detection: {anomaly['detection']}
- Timestamp: {anomaly['timestamp']}
- Date: {datetime.now().strftime('%Y-%m-%d')}

Alert Explanation:
"""
            
            if anomaly['type'] == 'Asset Detection':
                if anomaly['class'] in ['fish', 'fishing_pond']:
                    body += "- Fish detected in asset monitoring area (unexpected biological presence)"
                elif anomaly['class'] == 'major_shift':
                    body += "- Major structural shift detected (potential equipment displacement)"
                elif anomaly['class'] == 'moving_camera_stick':
                    body += "- Camera movement detected (potential equipment instability)"
            else:  # Fish Detection
                if anomaly['class'] in ['lying_oil_rig', 'lying_pipe']:
                    body += "- Infrastructure detected in fish monitoring area (unexpected asset presence)"
                elif anomaly['class'] == 'major_shift':
                    body += "- Major environmental shift in fish habitat"
            
            body += f"""

System Information:
- Monitoring Mode: {anomaly['type']}
- Alert Level: {anomaly['severity']}

Please investigate immediately if this is a CRITICAL alert.

---
Underwater Monitoring System
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
                
            print(f"Email alert sent for {anomaly['severity']} anomaly: {anomaly['class']}")
            
        except Exception as e:
            print(f"Failed to send email alert: {e}")
    
    def send_fish_drop_alert(self, anomaly, baseline_count, current_count):
        """Send specific email alert for fish count drop"""
        if not self.enabled:
            return False
        
        # Send fish drop email in background
        threading.Thread(target=self._send_fish_drop_email, args=(anomaly, baseline_count, current_count), daemon=True).start()
        return True
    
    def _send_fish_drop_email(self, anomaly, baseline_count, current_count):
        """Send fish drop email asynchronously"""
        try:
            subject = f"🐟 FISH COUNT DROP ALERT - Underwater Monitoring"
            
            drop_percentage = ((baseline_count - current_count) / baseline_count) * 100
            
            body = f"""
FISH COUNT DROP DETECTED

Alert Details:
- Severity: {anomaly['severity']}
- Detection Type: Fish Monitoring
- Alert Type: Fish Count Drop
- Timestamp: {anomaly['timestamp']}
- Date: {datetime.now().strftime('%Y-%m-%d')}

Fish Count Analysis:
- Baseline Count: {int(baseline_count)} fish
- Current Count: {int(current_count)} fish
- Drop Percentage: {drop_percentage:.1f}%
- Duration: Sustained for 10+ seconds

Possible Causes:
- Fish migration or movement
- Environmental disturbance
- Equipment malfunction
- Water quality changes
- Predator presence

Recommended Actions:
1. Check video feed for visual confirmation
2. Verify equipment is functioning properly
3. Monitor for recovery in fish count
4. Consider environmental factors

System Information:
- Monitoring Mode: Fish Detection
- Alert Threshold: 30% drop sustained for 10 seconds
- Detection Method: Sonar Signal Analysis

---
Underwater Fish Monitoring System
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
                
            print(f"Fish drop alert email sent: {int(baseline_count)} -> {int(current_count)}")
            
        except Exception as e:
            print(f"Failed to send fish drop email: {e}")
    
    def send_test_email(self):
        """Send test email to verify configuration"""
        if not self.enabled:
            return False, "Email not configured"
            
        test_anomaly = {
            'severity': 'HIGH',
            'type': 'Asset Detection',
            'class': 'test',
            'detection': 'test 0.95',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        
        try:
            self._send_email_async(test_anomaly)
            return True, "Test email sent successfully"
        except Exception as e:
            return False, f"Test email failed: {e}"