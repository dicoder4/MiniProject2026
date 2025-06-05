"""
Authentication Components for Flood Evacuation System
Handles login, user roles, and session management
"""

import streamlit as st
import hashlib
import json
import os
from datetime import datetime, timedelta

class AuthManager:
    def __init__(self):
        self.users_file = "users.json"
        self.session_timeout = 30  # minutes
        self.init_default_users()
    
    def init_default_users(self):
        """Initialize default users if file doesn't exist"""
        if not os.path.exists(self.users_file):
            default_users = {
                "admin": {
                    "password": self.hash_password("admin123"),
                    "role": "researcher",
                    "name": "System Administrator",
                    "created": datetime.now().isoformat()
                },
                "researcher": {
                    "password": self.hash_password("research123"),
                    "role": "researcher", 
                    "name": "Emergency Researcher",
                    "created": datetime.now().isoformat()
                },
                "citizen": {
                    "password": self.hash_password("citizen123"),
                    "role": "citizen",
                    "name": "Emergency Citizen",
                    "created": datetime.now().isoformat()
                }
            }
            self.save_users(default_users)
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def load_users(self):
        """Load users from JSON file"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_users(self, users):
        """Save users to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def authenticate(self, username, password):
        """Authenticate user credentials"""
        users = self.load_users()
        if username in users:
            hashed_password = self.hash_password(password)
            if users[username]["password"] == hashed_password:
                return users[username]
        return None
    
    def register_user(self, username, password, role, name):
        """Register new user"""
        users = self.load_users()
        if username in users:
            return False, "Username already exists"
        
        users[username] = {
            "password": self.hash_password(password),
            "role": role,
            "name": name,
            "created": datetime.now().isoformat()
        }
        self.save_users(users)
        return True, "User registered successfully"

def show_login_page():
    """Display login page"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>🌊 Emergency Flood Evacuation System</h1>
        <p style='color: #e8f4fd; margin: 0.5rem 0 0 0;'>Secure Access Portal</p>
    </div>
    """, unsafe_allow_html=True)
    
    auth_manager = AuthManager()
    
    # Create tabs for login and registration
    login_tab, register_tab, demo_tab = st.tabs(["🔐 Login", "📝 Register", "🎮 Demo Access"])
    
    with login_tab:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            remember_me = st.checkbox("Remember me for 30 days")
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("🔓 Login", type="primary", use_container_width=True)
            with col2:
                guest_button = st.form_submit_button("👤 Guest Access", use_container_width=True)
            
            if login_button:
                if username and password:
                    user_data = auth_manager.authenticate(username, password)
                    if user_data:
                        st.session_state.authenticated = True
                        st.session_state.user_role = user_data["role"]
                        st.session_state.user_name = user_data["name"]
                        st.session_state.username = username
                        st.session_state.login_time = datetime.now()
                        
                        if remember_me:
                            st.session_state.remember_login = True
                        
                        st.success(f"✅ Welcome back, {user_data['name']}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")
            
            if guest_button:
                st.session_state.authenticated = True
                st.session_state.user_role = "citizen"
                st.session_state.user_name = "Guest User"
                st.session_state.username = "guest"
                st.session_state.login_time = datetime.now()
                st.info("🎭 Logged in as Guest (Citizen Access)")
                st.rerun()
    
    with register_tab:
        st.subheader("Create New Account")
        
        with st.form("register_form"):
            new_username = st.text_input("Choose Username", placeholder="Enter desired username")
            new_name = st.text_input("Full Name", placeholder="Enter your full name")
            new_password = st.text_input("Password", type="password", placeholder="Enter password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
            role_choice = st.selectbox("Account Type", ["citizen", "researcher"], 
                                     format_func=lambda x: "🏠 Citizen (Emergency Access)" if x == "citizen" else "🔬 Researcher (Full Access)")
            
            register_button = st.form_submit_button("📝 Create Account", type="primary", use_container_width=True)
            
            if register_button:
                if all([new_username, new_name, new_password, confirm_password]):
                    if new_password == confirm_password:
                        if len(new_password) >= 6:
                            success, message = auth_manager.register_user(new_username, new_password, role_choice, new_name)
                            if success:
                                st.success(f"✅ {message}")
                                st.info("You can now login with your new account")
                            else:
                                st.error(f"❌ {message}")
                        else:
                            st.error("❌ Password must be at least 6 characters long")
                    else:
                        st.error("❌ Passwords do not match")
                else:
                    st.warning("⚠️ Please fill in all fields")
    
    with demo_tab:
        st.subheader("Demo Accounts")
        st.info("Use these demo accounts to explore the system:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔬 Researcher Demo**
            - Username: `researcher`
            - Password: `research123`
            - Access: Full system access
            """)
            
            if st.button("🔬 Login as Researcher", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.user_role = "researcher"
                st.session_state.user_name = "Demo Researcher"
                st.session_state.username = "researcher"
                st.session_state.login_time = datetime.now()
                st.rerun()
        
        with col2:
            st.markdown("""
            **🏠 Citizen Demo**
            - Username: `citizen`
            - Password: `citizen123`
            - Access: Emergency evacuation only
            """)
            
            if st.button("🏠 Login as Citizen", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.user_role = "citizen"
                st.session_state.user_name = "Demo Citizen"
                st.session_state.username = "citizen"
                st.session_state.login_time = datetime.now()
                st.rerun()

def check_authentication():
    """Check if user is authenticated and session is valid"""
    if 'authenticated' not in st.session_state:
        return False
    
    if not st.session_state.authenticated:
        return False
    
    # Check session timeout
    if 'login_time' in st.session_state:
        login_time = st.session_state.login_time
        if datetime.now() - login_time > timedelta(minutes=30):
            # Session expired
            st.session_state.authenticated = False
            st.warning("⏰ Session expired. Please login again.")
            return False
    
    return True

def show_user_info():
    """Display user information in sidebar"""
    if check_authentication():
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Information")
            
            role_icon = "🔬" if st.session_state.user_role == "researcher" else "🏠"
            role_name = "Researcher" if st.session_state.user_role == "researcher" else "Citizen"
            
            st.markdown(f"""
            **{role_icon} {st.session_state.user_name}**  
            Role: {role_name}  
            Username: {st.session_state.username}
            """)
            
            # Session info
            if 'login_time' in st.session_state:
                login_time = st.session_state.login_time
                session_duration = datetime.now() - login_time
                st.markdown(f"Session: {session_duration.seconds // 60} min")
            
            # Logout button
            if st.button("🚪 Logout", use_container_width=True):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

def require_role(required_role):
    """Decorator to require specific role for access"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not check_authentication():
                st.error("🔒 Please login to access this feature")
                return None
            
            if st.session_state.user_role != required_role and required_role != "any":
                st.error(f"🚫 This feature requires {required_role} access")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
