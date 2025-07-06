
import pytest
from emergency_notifications import EmergencyNotificationSystem

@pytest.fixture
def notification_system():
    return EmergencyNotificationSystem()

def test_send_email_alert(monkeypatch, notification_system):
    def mock_send_message(self, msg):
        print("Mock send_message called")
    
    monkeypatch.setattr("smtplib.SMTP.send_message", mock_send_message)
    monkeypatch.setattr("smtplib.SMTP.login", lambda self, user, pwd: None)
    monkeypatch.setattr("smtplib.SMTP.starttls", lambda self: None)
    monkeypatch.setattr("smtplib.SMTP.quit", lambda self: None)

    result = notification_system.send_email_alert("recipient@example.com", "Test Subject", "Test message")
    assert result is True

def test_send_sms_via_email(monkeypatch, notification_system):
    monkeypatch.setattr(notification_system, "send_email_alert", lambda *args, **kwargs: True)
    result = notification_system.send_sms_via_email("+911234567890", "Test SMS")
    assert result is True
