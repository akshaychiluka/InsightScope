import streamlit as st
import pyrebase
import firebase_admin
from firebase_admin import credentials, auth as admin_auth

# Firebase Setup
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\aksha\Downloads\insightscope-d0243-firebase-adminsdk-fbsvc-f8ff4b4ae5.json")
    firebase_admin.initialize_app(cred)

firebase_config = {
    "apiKey": "AIzaSyDpMmwYp4ViIFumeY_TZbattRvmnmsbnpQ",
    "authDomain": "insightscope-d0243.firebaseapp.com",
    "projectId": "insightscope-d0243",
    "storageBucket": "insightscope-d0243.firebasestorage.app",
    "messagingSenderId": "847688848408",
    "appId": "1:847688848408:web:ec065759c087c2c152df6d",
    "measurementId": "G-Z08EHPE0M8",
    "databaseURL": "https://insightscope-d0243.firebaseio.com"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

def sign_up(email, password, confirm_password):
    if not email or not password or not confirm_password:
        st.error("All fields are required!")
        return False
    if password != confirm_password:
        st.error("Passwords do not match!")
        return False
    
    try:
        user = auth.create_user_with_email_and_password(email, password)
        auth.send_email_verification(user['idToken'])
        st.success("Sign up successful! Check your email for verification.")
        st.info("Please verify your email before signing in.")
        return True
    except Exception as e:
        error_message = str(e)
        if "EMAIL_EXISTS" in error_message:
            st.error("Email already registered!")
        else:
            st.error(f"Sign up failed: {error_message}")
        return False

def sign_in(email, password):
    if not email or not password:
        st.error("Email and password are required!")
        return False
    
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        user_info = admin_auth.get_user_by_email(email)
        if not user_info.email_verified:
            st.error("Please verify your email first!")
            auth.send_email_verification(user['idToken'])
            st.info("Verification email resent.")
            return False
        st.session_state.user = user
        st.success("Signed in successfully!")
        return True
    except Exception as e:
        st.error(f"Sign in failed: {str(e)}")
        return False

def reset_password(email):
    if not email:
        st.error("Please enter your email to reset password!")
        return False
    try:
        auth.send_password_reset_email(email)
        st.success("Password reset email sent! Check your inbox.")
        return True
    except Exception as e:
        st.error(f"Failed to send reset email: {str(e)}")
        return False

def sign_out():
    st.session_state.user = None
    st.session_state.messages = [{"role": "assistant", "content": "You have been signed out. Please sign in to continue."}]
    st.session_state.history = []
    st.session_state.extracted_texts = []
    st.session_state.suggested_questions = []
    st.session_state.url_list = []
    st.session_state.last_query = None
    st.success("Signed out successfully!")

def show_auth_page():
    st.markdown("<h1 style='text-align: center; color: #ffffff;'>Welcome to InsightScope</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #aaaaaa;'>Sign in to explore AI-powered news research</p>", unsafe_allow_html=True)
    st.markdown("<div class='auth-form'>", unsafe_allow_html=True)
    
    # Tabs for Sign In and Sign Up
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
    with tab1:
        email = st.text_input("E-mail", placeholder="Enter your email", key="signin_email")
        password = st.text_input("Password", placeholder="Enter your password", type="password", key="signin_password")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Log In", key="signin_btn"):
                if sign_in(email, password):
                    st.rerun()
        with col2:
            if st.button("Forgot your password?", key="forgot_password_btn"):
                reset_password(email)

    with tab2:
        email = st.text_input("E-mail", placeholder="Enter your email", key="signup_email")
        password = st.text_input("Password", placeholder="Enter your password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", placeholder="Confirm your password", type="password", key="signup_confirm")
        if st.button("Sign Up", key="signup_btn"):
            sign_up(email, password, confirm_password)
    
    st.markdown("</div>", unsafe_allow_html=True)