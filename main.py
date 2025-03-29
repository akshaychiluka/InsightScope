import streamlit as st


st.set_page_config(page_title="InsightScope", page_icon="📈", layout="wide")

import os
from auth import show_auth_page
from ui import main_app

# Load CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please sign in to continue."}]
if "extracted_texts" not in st.session_state:
    st.session_state.extracted_texts = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "url_list" not in st.session_state:
    st.session_state.url_list = []
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "user" not in st.session_state:
    st.session_state.user = None

# App Entry Point
if not st.session_state.user:
    show_auth_page()
else:
    main_app()