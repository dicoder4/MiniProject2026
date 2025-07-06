
import pytest
from datetime import datetime
from auth_components import AuthManager

@pytest.fixture
def auth_manager():
    return AuthManager()

def test_register_user(auth_manager):
    username = "user5"
    password = "Test@1234"
    role = "researcher"
    name = "Test User"
    email = "testuser@example.com"
    phone = "+911234567899"

    success, message = auth_manager.register_user(username, password, role, name, email, phone)
    assert success or "exists" in message.lower()

def test_authenticate_user(auth_manager):
    user = auth_manager.authenticate("user5", "Test@1234")
    assert user is not None
    assert user["username"] == "user5"

def test_update_user_email_phone(auth_manager):
    updated = auth_manager.update_user_email_phone("user5", "updated@example.com", "+911111111111")
    assert updated is True
