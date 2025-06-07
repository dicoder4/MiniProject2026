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
from twilio.rest import Client

account_sid = 'AC4ec25d9275460e02e5f0cee617ef321b'
auth_token = 'd13337f28d0d6f56529d11bd4e2e81db'
twilio_number = '+16204481914'

# Create Twilio client
client = Client(account_sid, auth_token)



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
        
        # SMS Gateway mappings for Indian carriers
        self.sms_gateways = {
            # Airtel India
            'airtel': '@airtelmail.com',
            'airtel_ap': '@airtelap.com',
            'airtel_goa': '@airtelmail.com',
            'airtel_gujarat': '@airtelmail.com',
            'airtel_haryana': '@airtelmail.com',
            'airtel_himachal': '@airtelmail.com',
            'airtel_karnataka': '@airtelkk.com',
            'airtel_kerala': '@airtelkerala.com',
            'airtel_kolkata': '@airtelkol.com',
            'airtel_mp': '@airtelmail.com',
            'airtel_maharashtra': '@airtelmail.com',
            
            # Idea Cellular India
            'idea': '@ideacellular.net',
            'idea_ap': '@ideacellular.net',
            'idea_goa': '@ideacellular.net',
            'idea_gujarat': '@ideacellular.net',
            'idea_maharashtra': '@ideacellular.net',
            
            # Vodafone India
            'vodafone': '@vodafonemail.com',
            'vodafone_mumbai': '@vfmail.com',
            
            # Jio (Reliance)
            'jio': '@jiomsg.com',
            'reliance': '@rcom.co.in',
            
            # BSNL
            'bsnl': '@bsnlmail.com',
            
            # Other Indian carriers
            'tata_docomo': '@tatadocomo.com',
            'uninor': '@uninor.in',
            'mts': '@mtnl.net',
            'cellone': '@cellonemail.com',
            'spice': '@spicemobile.com',
            
            # Fallback options
            'default': '@airtelmail.com'  # Most widely used
        }
    
    def detect_indian_carrier(self, phone_number: str) -> str:
        """Detect Indian carrier based on phone number patterns"""
        # Remove all non-digits
        clean_phone = ''.join(filter(str.isdigit, phone_number))
        
        # Remove country code if present
        if len(clean_phone) == 12 and clean_phone.startswith('91'):
            clean_phone = clean_phone[2:]
        elif len(clean_phone) == 13 and clean_phone.startswith('091'):
            clean_phone = clean_phone[3:]
        
        if len(clean_phone) != 10:
            return 'airtel'  # Default fallback
        
        # Basic carrier detection based on number series
        # Note: These are approximate patterns and may not be 100% accurate
        first_digits = clean_phone[:4]
        
        # Airtel patterns (approximate)
        airtel_patterns = ['9876', '9877', '9878', '9879', '8447', '8448', '8449']
        if any(clean_phone.startswith(pattern[:3]) for pattern in airtel_patterns):
            return 'airtel'
        
        # Jio patterns (approximate)
        jio_patterns = ['8901', '8902', '8903', '8904', '8905', '6299', '7299', '7338']
        if any(clean_phone.startswith(pattern[:3]) for pattern in jio_patterns):
            return 'jio'
        
        # Vodafone patterns (approximate)
        vodafone_patterns = ['9825', '9826', '9827', '9828', '9829']
        if any(clean_phone.startswith(pattern[:3]) for pattern in vodafone_patterns):
            return 'vodafone'
        
        # Default to Airtel (most common)
        return 'airtel'
    
    def send_sms_via_email(self, phone_number: str, message: str, carrier: str = None) -> bool:
        """Send SMS via email-to-SMS gateway for Indian carriers"""
        try:
            # Clean phone number
            clean_phone = ''.join(filter(str.isdigit, phone_number))
            
            # Remove country code if present
            if len(clean_phone) == 12 and clean_phone.startswith('91'):
                clean_phone = clean_phone[2:]
            elif len(clean_phone) == 13 and clean_phone.startswith('091'):
                clean_phone = clean_phone[3:]
            
            if len(clean_phone) != 10:
                logger.error(f"Invalid Indian phone number format: {phone_number}")
                return False
            
            # Detect carrier if not provided
            if not carrier:
                carrier = self.detect_indian_carrier(phone_number)
            
            # Get SMS gateway
            gateway = self.sms_gateways.get(carrier.lower(), '@airtelmail.com')
            sms_email = clean_phone + gateway
            
            # Create SMS message (keep it short for SMS)
            sms_message = message[:160]  # SMS limit
            
            msg = MIMEText(sms_message)
            msg['From'] = self.alert_email
            msg['To'] = sms_email
            msg['Subject'] = "FLOOD ALERT"
            
            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.alert_email, self.alert_password)
                server.send_message(msg)
            
            logger.info(f"SMS sent successfully to {phone_number} via {sms_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS to {phone_number}: {e}")
            return False
    
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


# Updated Email message for user (bilingual with state-specific languages)
def get_flood_alert_email(user_name, state):
    
    STATE_LANGUAGE_MAP = {
        "Maharashtra": {
            "lang": "Marathi",
            "sms": "🚨 पूर इशारा FLOOD ALERT 🚨\n{name}, आपण पूर क्षेत्रात आहात!\nRoute: {algorithm}\nTime: {evacuation_time} min\nकृपया त्वरित स्थलांतर करा!\nEmergency: 112, 100, 108",
            "email_subject": "🚨 तातडीची सूचना: पूर स्थलांतर आवश्यक - Flood Evacuation Required",
            "email_greeting": "प्रिय {name},\n\nआपण सध्या पूर क्षेत्रात आहात. कृपया त्वरित स्थलांतर करा!\n\nDear {name},\nYou are currently in a flood zone and must evacuate immediately!"
        },
        "Karnataka": {
            "lang": "Kannada",
            "sms": "🚨 ನೆರೆ ಎಚ್ಚರಿಕೆ FLOOD ALERT 🚨\n{name}, ನೀವು ನೆರೆ ಪ್ರದೇಶದಲ್ಲಿದ್ದೀರಿ!\nRoute: {algorithm}\nTime: {evacuation_time} min\nದಯವಿಟ್ಟು ತಕ್ಷಣ ಸ್ಥಳಾಂತರಗೊಳ್ಳಿ!\nEmergency: 112, 100, 108",
            "email_subject": "🚨 ತುರ್ತು ಸೂಚನೆ: ನೆರೆ ಸ್ಥಳಾಂತರ ಅಗತ್ಯವಿದೆ - Flood Evacuation Required",
            "email_greeting": "ಪ್ರಿಯ {name},\n\nನೀವು ಈಗ ನೆರೆ ಪ್ರದೇಶದಲ್ಲಿದ್ದೀರಿ. ದಯವಿಟ್ಟು ತಕ್ಷಣ ಸ್ಥಳಾಂತರಗೊಳ್ಳಿ!\n\nDear {name},\nYou are currently in a flood zone and must evacuate immediately!"
        }
    }
    
    # Get language-specific content
    lang_config = STATE_LANGUAGE_MAP.get(state, STATE_LANGUAGE_MAP["Karnataka"])  # Default to Karnataka if state not found
    
    user_email_subject = lang_config["email_subject"]
    
    # Language-specific content based on state
    if state == "Maharashtra":
        # Marathi content
        header_local = "🚨 आपत्कालीन पूर स्थलांतर इशारा"
        emergency_instructions_header = "🆘 आपत्कालीन सूचना / Emergency Instructions:"
        emergency_contacts_header = "📞 आपत्कालीन संपर्क / Emergency Contacts:"
        
        emergency_services = [
            ("आपत्कालीन सेवा / Emergency Services", "112"),
            ("पोलिस / Police", "100"),
            ("वैद्यकीय आपत्काल / Medical Emergency", "108"),
            ("अग्निशमन विभाग / Fire Department", "101"),
            ("आपत्ती व्यवस्थापन / Disaster Management", "1078")
        ]
        
        instructions = [
            
            "अॅपमध्ये दाखवलेल्या स्थलांतर मार्गाचे तत्काळ पालन करा / Follow the evacuation route shown in the app immediately",
            "फक्त आवश्यक वस्तू घ्या / Take essential items only (documents, medications, water)",
            "शांत राहा आणि जलद पण सुरक्षितपणे पुढे जा / Stay calm and move quickly but safely",
            "समस्या असल्यास 112 वर कॉल करा / Call 112 if you encounter any problems",
            "अधिकार्‍यांनी सुरक्षित घोषित करेपर्यंत परत येऊ नका / Do not return until authorities declare it safe"
        ]

        
        warning_text = "⚠️ ही एक स्वयंचलित आपत्कालीन इशारा आहे. कृपया तत्काळ स्थलांतर करा."
        safe_centers = "खालील सुरक्षित केंद्रांपैकी कोणत्याहीकडे स्थलांतर करा / Evacuate to any of these safe centers:"
    else:  # Karnataka (Kannada)
        # Kannada content
        header_local = "🚨 ತುರ್ತು ನೆರೆ ಸ್ಥಳಾಂತರ ಎಚ್ಚರಿಕೆ"
        emergency_instructions_header = "🆘 ತುರ್ತು ಸೂಚನೆಗಳು / Emergency Instructions:"
        emergency_contacts_header = "📞 ತುರ್ತು ಸಂಪರ್ಕಗಳು / Emergency Contacts:"
        
        emergency_services = [
            ("ತುರ್ತು ಸೇವೆಗಳು / Emergency Services", "112"),
            ("ಪೊಲೀಸ್ / Police", "100"),
            ("ವೈದ್ಯಕೀಯ ತುರ್ತು / Medical Emergency", "108"),
            ("ಅಗ್ನಿಶಾಮಕ ವಿಭಾಗ / Fire Department", "101"),
            ("ವಿಪತ್ತು ನಿರ್ವಹಣೆ / Disaster Management", "1078")
        ]
        
        instructions = [
            
            "ಅಪ್ಲಿಕೇಶನ್‌ನಲ್ಲಿ ತೋರಿಸಿದ ಸ್ಥಳಾಂತರ ಮಾರ್ಗವನ್ನು ತಕ್ಷಣ ಅನುಸರಿಸಿ / Follow the evacuation route shown in the app immediately",
            "ಅಗತ್ಯ ವಸ್ತುಗಳನ್ನು ಮಾತ್ರ ತೆಗೆದುಕೊಳ್ಳಿ / Take essential items only (documents, medications, water)",
            "ಶಾಂತವಾಗಿರಿ ಮತ್ತು ವೇಗವಾಗಿ ಆದರೆ ಸುರಕ್ಷಿತವಾಗಿ ಚಲಿಸಿ / Stay calm and move quickly but safely",
            "ಸಮಸ್ಯೆ ಎದುರಾದರೆ 112 ಗೆ ಕರೆ ಮಾಡಿ / Call 112 if you encounter any problems",
            "ಅಧಿಕಾರಿಗಳು ಸುರಕ್ಷಿತ ಎಂದು ಘೋಷಿಸುವವರೆಗೆ ಹಿಂತಿರುಗಬೇಡಿ / Do not return until authorities declare it safe"
        ]

        
        warning_text = "⚠️ ಇದು ಸ್ವಯಂಚಾಲಿತ ತುರ್ತು ಎಚ್ಚರಿಕೆಯಾಗಿದೆ. ದಯವಿಟ್ಟು ತಕ್ಷಣ ಸ್ಥಳಾಂತರಗೊಳ್ಳಿ."
        safe_centers = "ಈ ಸುರಕ್ಷಿತ ಕೇಂದ್ರಗಳಲ್ಲಿ ಯಾವುದಕ್ಕೂ ಸ್ಥಳಾಂತರಗೊಳ್ಳಿ / Evacuate to any of these safe centers:"
    user_email_message = f"""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6;">
    <div style="background-color: #dc3545; color: white; padding: 20px; text-align: center;">
        <h1>{header_local}</h1>
        <h1>🚨 EMERGENCY FLOOD EVACUATION ALERT</h1>
    </div>
    
    <div style="padding: 20px;">
        <div style="background-color: #f8f9fa; padding: 15px; margin-bottom: 20px; border-left: 4px solid #dc3545;">
            {lang_config["email_greeting"].format(name=user_name)}
        </div>
        
        
        <h3>{emergency_instructions_header}</h3>
        <ol>
            {"".join(f"<li>{instruction}</li>" for instruction in instructions)}
        </ol>
        
        <h3>{emergency_contacts_header}</h3>
        <ul>
            {"".join(f"<li><strong>{service}:</strong> {number}</li>" for service, number in emergency_services)}
        </ul>
        
        <div style="background-color: #fff3cd; padding: 15px; margin: 20px 0; border-left: 4px solid #ffc107;">
            <strong>{warning_text}</strong><br>
            <strong>⚠️ This is an automated emergency alert. Please evacuate immediately.</strong>
        </div>
        <h1>{safe_centers}</h1>

    
    """
    
    return user_email_subject, user_email_message
STATE_SMS_MAP = {
    "Maharashtra": "🚨 पूर इशारा! खालील सुरक्षित केंद्रांपैकी कोणत्याहीकडे स्थलांतर करा. / Flood Alert! Evacuate to any of these safe centers.",
    "Karnataka": "🚨 ನೆರೆ ಎಚ್ಚರಿಕೆ! ಈ ಸುರಕ್ಷಿತ ಕೇಂದ್ರಗಳಲ್ಲಿ ಯಾವುದಕ್ಕೂ ಸ್ಥಳಾಂತರಗೊಳ್ಳಿ./ Flood Alert! Evacuate to any of these safe centers.",
    "Default": "🚨 Flood Alert! Evacuate to any of these safe centers."
}

def send_sms_alert_auth(user_name, user_ph, state):
    """Send a simple flood alert SMS in local language."""
    try:
        message_text = STATE_SMS_MAP.get(state, STATE_SMS_MAP["Default"])

        message = client.messages.create(
            body=message_text,
            from_=twilio_number,
            to=user_ph
        )

        print(f"✅ Message sent to {user_name} ({user_ph}) with SID: {message.sid}")
        return message.sid

    except Exception as e:
        print(f"❌ Failed to send SMS to {user_ph}: {e}")
        return None

# Test the connection when this module is imported (optional)
if __name__ == "__main__":
    test_gmail_connection()
