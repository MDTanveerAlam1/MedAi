import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import requests
import random
import io
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from openai import OpenAI

# ===================== PAGE CONFIGURATION =====================
st.set_page_config(
    page_title="MedGuide - Drug Recommender",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== DARK/LIGHT MODE TOGGLE =====================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # default to dark mode

# Toggle Button
with st.sidebar:
    toggle = st.button("🌙 TOGGLE DARK MODE" if not st.session_state.dark_mode else "☀️ TOGGLE LIGHT MODE")
    if toggle:
        st.session_state.dark_mode = not st.session_state.dark_mode

# Set Colors Based on Theme
if st.session_state.dark_mode:
    bg_color = "#0a192f"
    sidebar_bg = "#0a192f"
    text_color = "#ffffff"
    accent_color = "#00d084"
    input_bg = "#1e293b"
    border_color = "#38bdf8"
else:
    bg_color = "#ffffff"
    sidebar_bg = "#f0f2f6"
    text_color = "#000000"
    accent_color = "#228be6"
    input_bg = "#ffffff"
    border_color = "#cccccc"

# Inject Dynamic CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
    }}
    section[data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}
    section[data-testid="stSidebar"] a {{
        color: {accent_color} !important;
    }}
    section[data-testid="stSidebar"] .stButton>button {{
        background-color: {accent_color};
        color: {'black' if st.session_state.dark_mode else 'white'};
        font-weight: bold;
        border-radius: 10px;
        padding: 0.4rem 1rem;
    }}
    section[data-testid="stSidebar"] input {{
        background-color: {input_bg};
        color: {text_color};
        border: 1px solid {border_color};
    }}
    </style>
""", unsafe_allow_html=True)

# ===================== DATA AND MODEL LOADING =====================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("filter data.csv")
    except FileNotFoundError:
        st.error("❌ Dataset file 'filter data.csv' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

def load_model(model_path, encoder_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        if not hasattr(encoder, 'transform'):
            st.error("❌ Encoder lacks transform method.")
            return None, None
        return model, encoder
    except FileNotFoundError:
        st.error("❌ Model or encoder file not found.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model/encoder: {e}")
        return None, None

data = load_data()
model, encoder = load_model("model.pkl", "encoder.pkl")

# ===================== USER DATA UPLOAD & AI LEARNING (Persistent) =====================
USER_DATA_FILE = "user_uploaded_data.csv"
def save_user_data(df):
    df.to_csv(USER_DATA_FILE, index=False)

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        return pd.read_csv(USER_DATA_FILE)
    return pd.DataFrame(columns=["drugName", "condition", "review", "rating"])

if "user_learned_data" not in st.session_state:
    st.session_state["user_learned_data"] = load_user_data()

# ===================== CONTINUE WITH YOUR APP FUNCTIONALITY =====================
st.sidebar.markdown("""
### 🏠 Home
### 🧪 Predict Review
### 📊 Analytics
### 📤 Upload Data
### 💬 Chat with AI
### ℹ️ About
### 💊 Pharmacy Tools
""")

st.title("💊 MedGuide - Drug Recommender")
st.write("Welcome to **MedGuide**! Use the sidebar to navigate through prediction, analytics, uploads, and more.")

# Further app logic can be implemented here.
# Example: st.dataframe(data.head())
# Example: prediction input and output UI

