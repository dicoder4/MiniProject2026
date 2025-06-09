import requests
from twilio.rest import Client
import smtplib
import os
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from db_utils import get_users_collection

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
api_key = os.getenv("GOOGLE_MAPS_API_KEY")
twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
alert_email = os.getenv("ALERT_EMAIL")
alert_password = os.getenv("ALERT_PASSWORD")
smtp_server = os.getenv("SMTP_SERVER")
smtp_port = int(os.getenv("SMTP_PORT", 587))

# Create Twilio client
client = Client(account_sid, auth_token)

def find_nearest_safe_center_gmaps(user_address, safe_centers ):
    """
    user_address: str - Address of the user
    safe_centers: list of dicts with lat/lon
    api_key: str - Google Maps API key
    """
    # Geocode the user address
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={user_address}&key={api_key}"
    response = requests.get(geocode_url).json()

    if response["status"] != "OK" or not response["results"]:
        print("Failed to geocode user address.")
        return None

    user_location = response["results"][0]["geometry"]["location"]
    user_latlon = f"{user_location['lat']},{user_location['lng']}"
    print(user_location, user_latlon)

    # Build destinations string for all mock centers
    destinations = "|".join([f"{c['lat']},{c['lon']}" for c in safe_centers])

    # Call Distance Matrix API
    matrix_url = (
        f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={user_latlon}"
        f"&destinations={destinations}&key={api_key}"
    )
    matrix_response = requests.get(matrix_url).json()

    if matrix_response["status"] != "OK":
        print("Distance Matrix API error:", matrix_response.get("error_message"))
        return None

    distances = matrix_response["rows"][0]["elements"]

    # Find the closest center
    min_dist = float("inf")
    best_center = None
    for i, element in enumerate(distances):
        if element["status"] == "OK":
            print(f"Distance to center {i}: {element['distance']['text']}")
            dist_km = element["distance"]["value"] / 1000  # meters to km
            if dist_km < min_dist:
                min_dist = dist_km
                best_center = safe_centers[i]
                best_center["distance_km"] = dist_km
                best_center["gmaps_link"] = f"https://www.google.com/maps/search/?api=1&query={best_center['lat']},{best_center['lon']}"
    print(f"Closest center: {best_center} at {min_dist:.2f} km")
    return best_center


def send_email_alert(recipient_email: str, subject: str, message: str, is_html: bool = False) -> bool:
        """Send email alert"""
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = alert_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            if is_html:
                msg.attach(MIMEText(message, 'html'))
            else:
                msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(alert_email, alert_password)
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {recipient_email}: {e}")
            return False
        
STATE_SMS_MAP = {
    "Maharashtra": "🚨 पूर इशारा! आपत्कालीन क्रमांक: 100 (पोलीस), 108 (रुग्णवाहिका), 101 (अग्निशमन) / Flood Alert! Emergency numbers: 100 (Police), 108 (Ambulance), 101 (Fire)",
    "Karnataka": "🚨 ನೆರೆ ಎಚ್ಚರಿಕೆ! ತುರ್ತು ಸಂಖ್ಯೆಗಳು: 100 (ಪೊಲೀಸ್), 108 (ಆಂಬ್ಯುಲೆನ್ಸ್), 101 (ಅಗ್ನಿಶಾಮಕ) / Flood Alert! Emergency numbers: 100 (Police), 108 (Ambulance), 101 (Fire)",
    "Default": "🚨 Flood Alert! Emergency numbers: 100 (Police), 108 (Ambulance), 101 (Fire)"
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
    
    
def get_all_users():
    users_col = get_users_collection()
    return list(users_col.find({}))

def emergency_sos(users, state):
    user_emails = []
    user_phone = []
    # Accepts users as a list of dicts (from MongoDB)
    for user_data in users:
        email = user_data.get('email')
        phone = user_data.get('phone')
        address = user_data.get('address', '')
        name = user_data.get('name', user_data.get('username', ''))
        username = user_data.get('username', '')
        if email:
            user_emails.append({
                'email': email,
                'name': name,
                'username': username,
                'address': address
            })
        if phone:
            user_phone.append({
                'phone': phone,
                'name': name,
                'username': username,
                'address': address
            })
    if state == "Maharashtra":
         instructions = [
        "शांत राहा. घाबरू नका. / Stay calm. Do not panic.",
        "तत्काळ उंचीवर किंवा जवळच्या सुरक्षित केंद्रावर स्थलांतर करा. / Move to higher ground or nearest safe center immediately."]
         user_email_message = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
            <div style="background-color: #dc3545; color: white; padding: 20px; text-align: center;">
                <h1>{STATE_SMS_MAP['Maharashtra']}</h1>
                <h1>🚨 EMERGENCY FLOOD EVACUATION ALERT</h1>
            </div>
            <div style="padding: 20px;">
                <ol>
                    {"".join(f"<li>{instruction}</li>" for instruction in instructions)}
                </ol>
            </div>
        </body>
        </html>
        """
         for user in user_emails:
                send_email_alert(user['email'],"🚨 तातडीची सूचना: पूर स्थलांतर आवश्यक - Flood Evacuation Required",user_email_message, is_html=True)

         for user in user_phone:
                send_sms_alert_auth(user['name'], user['phone'], state)

    elif state == "Karnataka":
         instructions = [ "ಶಾಂತವಾಗಿರಿ. ಆತಂಕಪಡಬೇಡಿ. / Stay calm. Do not panic.",
           "ತಕ್ಷಣವೇ ಎತ್ತರದ ಭೂಮಿಗೆ ಅಥವಾ ಸಮೀಪದ ಸುರಕ್ಷಿತ ಕೇಂದ್ರಕ್ಕೆ ಹೋಗಿ. / Move to higher ground or nearest safe center immediately."]
         user_email_message = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
            <div style="background-color: #dc3545; color: white; padding: 20px; text-align: center;">
                <h1>{STATE_SMS_MAP['Karnataka']}</h1>
                <h1>🚨 EMERGENCY FLOOD EVACUATION ALERT</h1>
            </div>
            <div style="padding: 20px;">
                <ol>
                    {"".join(f"<li>{instruction}</li>" for instruction in instructions)}
                </ol>
            </div>
        </body>
        </html>
        """
         for user in user_emails:
                send_email_alert(user['email'], "🚨 ತುರ್ತು ಸೂಚನೆ: ನೆರೆ ಸ್ಥಳಾಂತರ ಅಗತ್ಯವಿದೆ - Flood Evacuation Required", user_email_message, is_html=True)
         for user in user_phone:
                send_sms_alert_auth(user['name'], user['phone'], state)
                

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
            "शांत राहा. घाबरू नका. / Stay calm. Do not panic.",
            "तत्काळ उंचीवर किंवा जवळच्या सुरक्षित केंद्रावर स्थलांतर करा. / Move to higher ground or nearest safe center immediately.",
            "पूर पाण्यात चालू किंवा गाडी चालवू नका. फक्त ६ इंच पाणी तुम्हाला वाहून नेऊ शकते. / Do not walk or drive through flood water. Just 6 inches of water can sweep you away.",
            "आवश्यक वस्तू सोबत घ्या: ओळखपत्र, औषधे, पाणी, कोरडे अन्न, टॉर्च, फोन, चार्जर. / Carry essentials: ID, medicines, water, dry food, flashlight, phone, charger.",
            "पूर आलेल्या इमारतींमध्ये लिफ्ट/एलेव्हेटर वापरणे टाळा. / Avoid using lifts/elevators in flooded buildings.",
            "सुरक्षित असल्यास निघण्यापूर्वी वीज आणि गॅस बंद करा. / Turn off electricity and gas before leaving, if safe to do so.",
            "स्थानिक अधिकार्‍यांच्या सूचनांचे पालन करा. / Follow local authorities' instructions.",
            "वृद्ध, लहान मुलं आणि अपंगांची मदत करा. / Help the elderly, children, and people with disabilities.",
            "अधिकृत अपडेट्स रेडिओ, टीव्ही किंवा खात्रीशीर व्हॉट्सअॅप चॅनेलवरून ऐका. / Stay informed: Listen to official updates via radio, TV, or verified WhatsApp channels.",
            "अफवा किंवा अनधिकृत सोशल मीडिया बातम्या टाळा. / Avoid rumors or unverified social media news."
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
            "ಶಾಂತವಾಗಿರಿ. ಆತಂಕಪಡಬೇಡಿ. / Stay calm. Do not panic.",
            "ತಕ್ಷಣವೇ ಎತ್ತರದ ಭೂಮಿಗೆ ಅಥವಾ ಸಮೀಪದ ಸುರಕ್ಷಿತ ಕೇಂದ್ರಕ್ಕೆ ಹೋಗಿ. / Move to higher ground or nearest safe center immediately.",
            "ಮಳೆ ನೀರಿನಲ್ಲಿ ನಡೆದು ಹೋಗಬೇಡಿ ಅಥವಾ ವಾಹನ ಚಾಲನೆ ಮಾಡಬೇಡಿ. ಕೇವಲ 6 ಇಂಚು ನೀರು ನಿಮ್ಮನ್ನು ತಳ್ಳಿಬಿಡಬಹುದು. / Do not walk or drive through flood water. Just 6 inches of water can sweep you away.",
            "ಅವಶ್ಯಕ ವಸ್ತುಗಳನ್ನು ತೆಗೆದುಕೊಂಡು ಹೋಗಿ: ಗುರುತು ಪತ್ರ, ಔಷಧಿ, ನೀರು, ಒಣ ಆಹಾರ, ದೀಪ, ಫೋನ್, ಚಾರ್ಜರ್. / Carry essentials: ID, medicines, water, dry food, flashlight, phone, charger.",
            "ನೀರಿನಿಂದ ತುಂಬಿದ ಕಟ್ಟಡಗಳಲ್ಲಿ ಲಿಫ್ಟ್/ಎಲಿವೇಟರ್ ಬಳಕೆ ತಪ್ಪಿಸಿ. / Avoid using lifts/elevators in flooded buildings.",
            "ಸುರಕ್ಷಿತವಿದ್ದರೆ ಹೊರಟ ಮೊದಲೇ ವಿದ್ಯುತ್ ಮತ್ತು ಅನಿಲವನ್ನು ನಿಲ್ಲಿಸಿ. / Turn off electricity and gas before leaving, if safe to do so.",
            "ಸ್ಥಳೀಯ ಅಧಿಕಾರಿಗಳ ಸೂಚನೆಗಳನ್ನು ಅನುಸರಿಸಿ. / Follow local authorities' instructions.",
            "ವೃದ್ಧರು, ಮಕ್ಕಳ ಮತ್ತು ಅಶಕ್ತರ ಸಹಾಯ ಮಾಡಿ. / Help the elderly, children, and people with disabilities.",
            "ಸರ್ಕಾರಿ ನ್ಯೂಸ್‌ಗಳನ್ನು ರೇಡಿಯೋ, ಟಿವಿ ಅಥವಾ ಮಾನ್ಯವಾದ ವಾಟ್ಸ್‌ಆಪ್ ಚಾನೆಲ್‌ಗಳಿಂದ ಕೇಳಿ. / Stay informed: Listen to official updates via radio, TV, or verified WhatsApp channels.",
            "ಅನಿರೀಕ್ಷಿತ ಅಥವಾ ಅವಿಶ್ವಾಸಾರ್ಹ ಸಾಮಾಜಿಕ ಮಾಧ್ಯಮ ಸುದ್ದಿಗಳನ್ನು ತಪ್ಪಿಸಿ. / Avoid rumors or unverified social media news."
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