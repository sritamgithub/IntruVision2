import yagmail
import os
from datetime import datetime

class AlertSystem:
    def __init__(self, db):
        """Initialize the alert system"""
        self.db = db
        self.email_recipient = self.db.get_setting('email_recipient')
        self.alert_for_unknown = self.db.get_setting('alert_for_unknown') == 'true'
    
    def send_email_alert(self, detection_id, image_path, person_name="Unknown"):
        """Send an email alert for a face detection"""
        try:
            # Get recipient from database
            email_recipient = self.db.get_setting('email_recipient')
            
            if not email_recipient:
                return False, "No email recipient configured"
            
            # Setup email
            yag = yagmail.SMTP("autoemailsender2@gmail.com", "tczewxnxfrpviped")

            # Current timestamp
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Email subject
            subject = f"Face Recognition Alert: {person_name} detected at {now}"
            
            # Email body
            contents = [
                f"A face detection alert has been triggered at {now}.",
                f"Person identified: {person_name}",
                f"Please check the attached image.",
            ]
            
            # Add image attachment if available
            attachments = []
            if image_path and os.path.exists(image_path):
                attachments.append(image_path)
            
            # Send email
            yag.send(
                to=email_recipient,
                subject=subject,
                contents=contents,
                attachments=attachments
            )
            
            print("send")
            # Log alert
            self.db.log_alert(detection_id, 'email', 'sent')
            
            return True, "Email alert sent successfully"
            
        except Exception as e:
            print(e)
            # Log alert failure
            self.db.log_alert(detection_id, 'email', f'failed: {str(e)}')
            return False, str(e)
    
    def handle_detection(self, person_id, person_name, status, image_path):
        """Handle a face detection event and send alerts if needed"""
        # Log the detection
        detection_id = self.db.log_detection(person_id, status, image_path)
        
        # Check if we should send an alert
        should_alert = False
        
        if status == 'unknown' and self.alert_for_unknown:
            should_alert = True
        
        # Send alert if needed
        if should_alert:
            success, message = self.send_email_alert(detection_id, image_path, person_name)
            return success, message
        
        return True, "Detection logged (no alert needed)"