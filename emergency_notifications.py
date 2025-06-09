"""
Emergency Notification System for Flood Evacuation - India Edition
Sends SOS alerts via SMS (email-to-SMS) and email notifications
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging
from typing import List, Dict, Optional
from db_utils import get_users_collection


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class EmergencyNotificationSystem:
    def __init__(self):
        """Initialize the notification system with environment variables"""
        self.alert_email = os.getenv("ALERT_EMAIL", "floodevacuationsystem@gmail.com")
        self.alert_password = os.getenv("ALERT_EMAIL_PASSWORD", "kdxo hvhj hmcp glio")
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
    def send_email_alert(self, recipient_email: str, subject: str, message: str, is_html: bool = False) -> bool:
        """Send email alert"""
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.alert_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            if is_html:
                msg.attach(MIMEText(message, 'html'))
            else:
                msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.alert_email, self.alert_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {recipient_email}: {e}")
            return False
    
    def send_sms_via_email(self, phone_number: str, message: str) -> bool:
        """Send SMS via email-to-SMS gateway"""
        try:
            # Indian SMS gateways
            carriers = {
                "airtel": "@airtelap.com",
                "jio": "@jionet.co.in", 
                "vi": "@vtext.com",
                "bsnl": "@bsnlnet.in"
            }
            
            # Try multiple carriers
            for carrier, gateway in carriers.items():
                sms_email = phone_number.replace("+91", "") + gateway
                success = self.send_email_alert(sms_email, "Flood Alert", message)
                if success:
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to send SMS to {phone_number}: {e}")
            return False

# MOVE THESE FUNCTIONS OUTSIDE THE CLASS - AT MODULE LEVEL

def get_all_authorities():
    """Get all users with authority role from MongoDB"""
    try:
        from db_utils import get_all_users
        users = get_all_users()
        authorities = []
        for user in users:
            if user.get('role') == 'authority':
                authorities.append({
                    'username': user.get('username', ''),
                    'name': user.get('name', 'Unknown'),
                    'email': user.get('email', ''),
                    'phone': user.get('phone', '')
                })
        return authorities
    except Exception as e:
        logger.error(f"Failed to load authorities from MongoDB: {e}")
        return []

def send_evacuation_plan_to_authorities(researcher_data: Dict, evacuation_data: Dict, location_data: Dict, map_image_base64: str = None) -> Dict:
    """Send evacuation plan to all registered authorities with map"""
    results = {
        'authorities_notified': [],
        'failed_notifications': [],
        'total_sent': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Get all authorities
        authorities = get_all_authorities()
        
        if not authorities:
            logger.warning("No authorities found in system")
            return results
        
        # Extract data
        researcher_name = researcher_data.get('name', 'Unknown Researcher')
        researcher_email = researcher_data.get('email', '')
        location_name = location_data.get('location_name', 'Unknown Location')
        station_name = location_data.get('station_name', 'Unknown Station')
        lat = location_data.get('lat', 0)
        lon = location_data.get('lon', 0)
        
        algorithm = evacuation_data.get('algorithm', 'Unknown')
        evacuation_time = evacuation_data.get('evacuation_time', 0)
        evacuated_count = evacuation_data.get('evacuated_count', 0)
        total_at_risk = evacuation_data.get('total_at_risk', 0)
        success_rate = (evacuated_count / total_at_risk * 100) if total_at_risk > 0 else 0
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        
        # Create professional subject line
        subject = f"🚨 OFFICIAL FLOOD EVACUATION PLAN - {station_name} | Research Report by {researcher_name}"
        
        # Create comprehensive evacuation report
        email_message = f"""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 25px; text-align: center; border-radius: 10px 10px 0 0;">
        <h1 style="margin: 0; font-size: 24px;">🌊 OFFICIAL FLOOD EVACUATION PLAN</h1>
        <h2 style="margin: 10px 0 0 0; font-size: 18px; font-weight: normal;">Emergency Response Research Report</h2>
    </div>
    
    <div style="padding: 30px; background: #f8f9fa; border-radius: 0 0 10px 10px;">
        <div style="background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            
            <h2 style="color: #1e3c72; border-bottom: 2px solid #1e3c72; padding-bottom: 10px;">📋 EXECUTIVE SUMMARY</h2>
            
            <div style="background: #e3f2fd; padding: 15px; border-left: 4px solid #2196f3; margin: 20px 0;">
                <p><strong>Prepared by:</strong> {researcher_name} (Emergency Response Researcher)</p>
                <p><strong>Research Institution:</strong> Flood Evacuation Planning System</p>
                <p><strong>Report Date:</strong> {timestamp}</p>
                <p><strong>Location:</strong> {location_name}</p>
                <p><strong>Station:</strong> {station_name}</p>
            </div>
            
            <h3 style="color: #d32f2f;">🎯 CRITICAL FINDINGS</h3>
            <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                <tr style="background: #f5f5f5;">
                    <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Optimal Algorithm</td>
                    <td style="padding: 12px; border: 1px solid #ddd;">{algorithm}</td>
                </tr>
                <tr>
                    <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Average Evacuation Time</td>
                    <td style="padding: 12px; border: 1px solid #ddd;">{evacuation_time:.1f} minutes</td>
                </tr>
                <tr style="background: #f5f5f5;">
                    <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Success Rate</td>
                    <td style="padding: 12px; border: 1px solid #ddd;">{success_rate:.1f}% ({evacuated_count}/{total_at_risk} people)</td>
                </tr>
                <tr>
                    <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">Coordinates</td>
                    <td style="padding: 12px; border: 1px solid #ddd;">{lat:.6f}, {lon:.6f}</td>
                </tr>
            </table>
            
            <h3 style="color: #d32f2f;">🗺️ EVACUATION ROUTE MAP</h3>
            <div style="text-align: center; margin: 20px 0;">
                {"<img src='data:image/png;base64," + map_image_base64 + "' style='max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 8px;' alt='Evacuation Route Map'/>" if map_image_base64 else "<p style='color: #666; font-style: italic;'>Map visualization not available</p>"}
            </div>
            
            <h3 style="color: #d32f2f;">📊 RECOMMENDED ACTIONS</h3>
            <ol style="padding-left: 20px;">
                <li><strong>Immediate Deployment:</strong> Implement the {algorithm} evacuation algorithm for optimal results</li>
                <li><strong>Resource Allocation:</strong> Position emergency vehicles along identified evacuation routes</li>
                <li><strong>Communication:</strong> Alert residents in the affected area using the provided coordinates</li>
                <li><strong>Monitoring:</strong> Establish checkpoints at safe centers to track evacuation progress</li>
                <li><strong>Coordination:</strong> Liaise with local emergency services for seamless execution</li>
            </ol>
            
            <div style="background: #d4edda; padding: 20px; border-left: 4px solid #28a745; margin: 25px 0;">
                <h4 style="margin-top: 0; color: #155724;">📧 RESEARCHER CONTACT INFORMATION</h4>
                <p style="margin: 5px 0;"><strong>Lead Researcher:</strong> {researcher_name}</p>
                <p style="margin: 5px 0;"><strong>Email:</strong> {researcher_email}</p>
                <p style="margin: 5px 0;"><strong>Institution:</strong> Emergency Response Research Division</p>
                <p style="margin: 5px 0;"><strong>Report ID:</strong> FERP-{timestamp.replace(' ', '').replace(':', '').replace('-', '')}</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 2px solid #eee;">
                <p style="font-size: 12px; color: #666; margin: 0;">
                    <strong>CONFIDENTIAL GOVERNMENT COMMUNICATION</strong><br>
                    This evacuation plan is generated using advanced AI algorithms and real-time flood simulation data.<br>
                    For immediate assistance or clarification, contact the research team or emergency services.
                </p>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        # Initialize notification system
        notification_system = EmergencyNotificationSystem()
        
        # Send to all authorities
        for authority in authorities:
            if authority['email']:
                try:
                    success = notification_system.send_email_alert(
                        authority['email'], 
                        subject, 
                        email_message, 
                        is_html=True
                    )
                    
                    if success:
                        results['authorities_notified'].append({
                            'name': authority['name'],
                            'email': authority['email'],
                            'username': authority['username']
                        })
                        results['total_sent'] += 1
                    else:
                        results['failed_notifications'].append({
                            'name': authority['name'],
                            'email': authority['email'],
                            'error': 'Email send failed'
                        })
                        
                except Exception as e:
                    results['failed_notifications'].append({
                        'name': authority['name'],
                        'email': authority['email'],
                        'error': str(e)
                    })
            else:
                results['failed_notifications'].append({
                    'name': authority['name'],
                    'email': 'No email provided',
                    'error': 'Missing email address'
                })
        
        logger.info(f"Evacuation plan sent to {results['total_sent']} authorities")
        return results
        
    except Exception as e:
        logger.error(f"Failed to send evacuation plan to authorities: {e}")
        results['failed_notifications'].append({'error': str(e)})
        return results
   

    
    
    def send_email_alert(self, recipient_email: str, subject: str, message: str, is_html: bool = False) -> bool:
        """Send email alert"""
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.alert_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            if is_html:
                msg.attach(MIMEText(message, 'html'))
            else:
                msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.alert_email, self.alert_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {recipient_email}: {e}")
            return False
    
    def send_sos_alert(self, user_data: Dict, evacuation_data: Dict, location_data: Dict) -> Dict:
        """Send comprehensive SOS alert to user and authorities"""
        results = {
            'user_sms': False,
            'user_email': False,
            'authority_email': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract user information
        user_name = user_data.get('name', 'Unknown User')
        user_email = user_data.get('email', '')
        user_phone = user_data.get('phone', '')
        user_lat = location_data.get('lat', 0)
        user_lon = location_data.get('lon', 0)
        
        # Extract evacuation information
        algorithm = evacuation_data.get('best_algorithm', 'Unknown')
        evacuation_time = evacuation_data.get('best_time', 0)
        destination = evacuation_data.get('destination', 'Safe Center')
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        
        # SMS message for user (Hindi + English for India)
        sms_message = f"""🚨 बाढ़ चेतावनी FLOOD ALERT 🚨
{user_name}, आप बाढ़ क्षेत्र में हैं!
Route: {algorithm}
Time: {evacuation_time:.0f} min
तुरंत निकासी मार्ग का पालन करें!
Emergency: 112, 100, 108"""
        
        # Email message for user (bilingual)
        user_email_subject = "🚨 URGENT: बाढ़ निकासी आवश्यक - Flood Evacuation Required"
        user_email_message = f"""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6;">
    <div style="background-color: #dc3545; color: white; padding: 20px; text-align: center;">
        <h1>🚨 आपातकालीन बाढ़ निकासी चेतावनी</h1>
        <h1>🚨 EMERGENCY FLOOD EVACUATION ALERT</h1>
    </div>
    
    <div style="padding: 20px;">
        <h2>प्रिय {user_name} / Dear {user_name},</h2>
        
        <p><strong style="color: red;">आप वर्तमान में बाढ़ क्षेत्र में हैं और तुरंत निकासी करनी चाहिए!</strong></p>
        <p><strong style="color: red;">You are currently in a flood zone and must evacuate immediately!</strong></p>
        
        <h3>📍 आपका स्थान / Your Location:</h3>
        <ul>
            <li><strong>निर्देशांक / Coordinates:</strong> {user_lat:.6f}, {user_lon:.6f}</li>
            <li><strong>चेतावनी समय / Alert Time:</strong> {timestamp}</li>
        </ul>
        
        <h3>🚶 निकासी विवरण / Evacuation Details:</h3>
        <ul>
            <li><strong>सर्वोत्तम मार्ग / Best Route Algorithm:</strong> {algorithm}</li>
            <li><strong>अनुमानित समय / Estimated Time:</strong> {evacuation_time:.0f} minutes</li>
            <li><strong>गंतव्य / Destination:</strong> {destination}</li>
        </ul>
        
        <h3>🆘 आपातकालीन निर्देश / Emergency Instructions:</h3>
        <ol>
            <li>ऐप में दिखाए गए निकासी मार्ग का तुरंत पालन करें / Follow the evacuation route shown in the app immediately</li>
            <li>केवल आवश्यक वस्तुएं लें / Take essential items only (documents, medications, water)</li>
            <li>शांत रहें और तेज़ी से लेकिन सुरक्षित रूप से आगे बढ़ें / Stay calm and move quickly but safely</li>
            <li>समस्या होने पर 112 पर कॉल करें / Call 112 if you encounter any problems</li>
            <li>अधिकारियों द्वारा सुरक्षित घोषित करने तक वापस न लौटें / Do not return until authorities declare it safe</li>
        </ol>
        
        <h3>📞 आपातकालीन संपर्क / Emergency Contacts:</h3>
        <ul>
            <li><strong>आपातकालीन सेवाएं / Emergency Services:</strong> 112</li>
            <li><strong>पुलिस / Police:</strong> 100</li>
            <li><strong>चिकित्सा आपातकाल / Medical Emergency:</strong> 108</li>
            <li><strong>अग्निशमन विभाग / Fire Department:</strong> 101</li>
            <li><strong>आपदा प्रबंधन / Disaster Management:</strong> 1078</li>
        </ul>
        
        <div style="background-color: #fff3cd; padding: 15px; margin: 20px 0; border-left: 4px solid #ffc107;">
            <strong>⚠️ यह एक स्वचालित आपातकालीन चेतावनी है। कृपया तुरंत निकासी करें।</strong><br>
            <strong>⚠️ This is an automated emergency alert. Please evacuate immediately.</strong>
        </div>
    </div>
</body>
</html>
        """
        
        # Authority notification email
        authority_subject = f"🚨 FLOOD EVACUATION SOS - {user_name} at {user_lat:.4f}, {user_lon:.4f}"
        authority_message = f"""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6;">
    <div style="background-color: #dc3545; color: white; padding: 20px; text-align: center;">
        <h1>🚨 FLOOD EVACUATION SOS ALERT - INDIA</h1>
    </div>
    
    <div style="padding: 20px;">
        <h2>Emergency Evacuation Alert</h2>
        
        <h3>👤 Affected Person:</h3>
        <ul>
            <li><strong>Name:</strong> {user_name}</li>
            <li><strong>Email:</strong> {user_email}</li>
            <li><strong>Phone:</strong> {user_phone}</li>
        </ul>
        
        <h3>📍 Location Details:</h3>
        <ul>
            <li><strong>Coordinates:</strong> {user_lat:.6f}, {user_lon:.6f}</li>
            <li><strong>Alert Timestamp:</strong> {timestamp}</li>
            <li><strong>Google Maps:</strong> <a href="https://maps.google.com/?q={user_lat},{user_lon}">View Location</a></li>
        </ul>
        
        <h3>🚶 Evacuation Status:</h3>
        <ul>
            <li><strong>Algorithm Used:</strong> {algorithm}</li>
            <li><strong>Evacuation Time:</strong> {evacuation_time:.0f} minutes</li>
            <li><strong>Destination:</strong> {destination}</li>
            <li><strong>Status:</strong> SOS Alert Triggered</li>
        </ul>
        
        <h3>🚨 Required Actions:</h3>
        <ol>
            <li>Verify person's safety and evacuation status</li>
            <li>Dispatch emergency services if needed (NDRF/SDRF)</li>
            <li>Monitor evacuation progress</li>
            <li>Coordinate with local disaster management authorities</li>
            <li>Provide additional assistance if required</li>
        </ol>
        
        <h3>📞 Indian Emergency Contacts:</h3>
        <ul>
            <li><strong>Emergency Services:</strong> 112</li>
            <li><strong>Police:</strong> 100</li>
            <li><strong>Medical Emergency:</strong> 108</li>
            <li><strong>Fire Department:</strong> 101</li>
            <li><strong>Disaster Management:</strong> 1078</li>
            <li><strong>NDRF:</strong> 011-24363260</li>
        </ul>
        
        <div style="background-color: #f8d7da; padding: 15px; margin: 20px 0; border-left: 4px solid #dc3545;">
            <strong>⚠️ This is an automated SOS alert from the Flood Evacuation System. Immediate attention required.</strong>
        </div>
    </div>
</body>
</html>
        """
        
        # Send notifications
        if user_phone:
            results['user_sms'] = self.send_sms_via_email(user_phone, sms_message)
        
        if user_email:
            results['user_email'] = self.send_email_alert(user_email, user_email_subject, user_email_message, is_html=True)
        
        # Send to authorities
        results['authority_email'] = self.send_email_alert(
            self.alert_email, 
            authority_subject, 
            authority_message, 
            is_html=True
        )
        
        return results
    
    def send_evacuation_plan(self, user_data: Dict, evacuation_plan: Dict) -> bool:
        """Send detailed evacuation plan to user - THIS WAS MISSING!"""
        try:
            user_email = user_data.get('email', '')
            user_phone = user_data.get('phone', '')
            user_name = user_data.get('name', 'User')
            
            if not user_email and not user_phone:
                return False
            
            # Create evacuation plan message
            plan_details = evacuation_plan.get('details', 'No details available')
            route_map_url = evacuation_plan.get('map_url', '#')
            
            subject = f"📋 Your Flood Evacuation Plan - {user_name}"
            
            email_message = f"""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6;">
    <div style="background-color: #007bff; color: white; padding: 20px; text-align: center;">
        <h1>📋 Your Personalized Evacuation Plan</h1>
    </div>
    
    <div style="padding: 20px;">
        <h2>Dear {user_name},</h2>
        
        <p>Here is your personalized flood evacuation plan:</p>
        
        <div style="background-color: #e7f3ff; padding: 15px; margin: 20px 0; border-left: 4px solid #007bff;">
            <h3>📍 Evacuation Details:</h3>
            {plan_details}
        </div>
        
        <h3>📱 Keep This Information Handy:</h3>
        <ul>
            <li>Save this email for offline access</li>
            <li>Screenshot the evacuation route</li>
            <li>Share with family members</li>
            <li>Keep emergency contacts ready</li>
        </ul>
        
        <p><strong>Stay safe and be prepared!</strong></p>
    </div>
</body>
</html>
            """
            
            # Send email
            success = self.send_email_alert(user_email, subject, email_message, is_html=True)
            
            # Send SMS summary if phone available
            if user_phone and success:
                sms_summary = f"📋 Evacuation plan sent to your email ({user_email}). Check your inbox for detailed instructions. Stay prepared! - Flood Alert System"
                self.send_sms_via_email(user_phone, sms_summary)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send evacuation plan: {e}")
            return False

# Initialize global notification system
notification_system = EmergencyNotificationSystem()

def send_sos_alert(user_data: Dict, evacuation_data: Dict, location_data: Dict) -> Dict:
    """Wrapper function to send SOS alert"""
    return notification_system.send_sos_alert(user_data, evacuation_data, location_data)

def send_evacuation_plan(user_data: Dict, evacuation_plan: Dict) -> bool:
    """Wrapper function to send evacuation plan"""
    return notification_system.send_evacuation_plan(user_data, evacuation_plan)
# Add this at the very end of emergency_notifications.py, after the existing wrapper functions

def test_gmail_connection():
    """Test Gmail SMTP connection"""
    try:
        import smtplib
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('floodevacuationsystem@gmail.com', 'kdxo hvhj hmcp glio')
        server.quit()
        print("✅ Gmail connection successful!")
        return True
    except Exception as e:
        print(f"❌ Gmail connection failed: {e}")
        return False

def debug_user_data_session():
    """Debug function to check user data in session state"""
    import streamlit as st
    
    st.write("### 🔍 Debug User Data")
    st.write("**Session State:**")
    user_data = {
        'name': st.session_state.get('user_name', 'NOT SET'),
        'email': st.session_state.get('user_email', 'NOT SET'),
        'phone': st.session_state.get('user_phone', 'NOT SET'),
        'username': st.session_state.get('username', 'NOT SET')
    }
    st.json(user_data)
    
    # Check if values are empty strings
    for key, value in user_data.items():
        if not value or value == 'NOT SET':
            st.error(f"❌ {key} is missing!")
        else:
            st.success(f"✅ {key}: {value}")


# Test the connection when this module is imported (optional)
if __name__ == "__main__":
    test_gmail_connection()
