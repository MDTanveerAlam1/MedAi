'''import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="MedGuide - Drug Recommender",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dataset with caching and error handling
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

# Load model and encoder
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

# Load assets
data = load_data()
model, encoder = load_model("model.pkl", "encoder.pkl")

# Custom CSS for advanced UI
st.markdown("""
    <style>
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #0a192f 0%, #2a5298 100%);
        font-family: 'Inter', sans-serif;
        color: #e6e6fa;
        transition: background 0.3s ease;
    }

    /* Dark mode toggle */
    .dark-mode .stApp {
        background: linear-gradient(135deg, #1c2526 0%, #3b4a6b 100%);
    }

    h1 {
        background: linear-gradient(90deg, #00d4ff 0%, #90ee90 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 1.5rem;
    }

    h2, h3 {
        color: #00d4ff;
        font-weight: 600;
    }

    /* Card Styling */
    .drug-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .drug-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 32px rgba(0, 212, 255, 0.3);
    }

    /* Input Styling */
    .stTextInput > div > div > input, .stTextArea textarea, .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        color: #e6e6fa;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .stTextInput > div > div > input:focus, .stTextArea textarea:focus, .stSelectbox > div > div:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 8px rgba(0, 212, 255, 0.3);
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #90ee90 100%);
        color: #0a192f;
        font-weight: 700;
        border-radius: 12px;
        padding: 12px 24px;
        border: none;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(10, 25, 47, 0.95);
        backdrop-filter: blur(10px);
        padding: 20px;
        width: 280px;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stSidebar .stRadio label {
        font-size: 16px;
        color: #e6e6fa;
        padding: 10px;
        border-radius: 8px;
        transition: background 0.2s ease;
    }

    .stSidebar .stRadio label:hover {
        background: rgba(0, 212, 255, 0.1);
    }

    /* Review Container Styling */
    .review-container {
        max-height: 300px;
        overflow-y: auto;
        background: rgba(0, 212, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        scrollbar-width: thin;
        scrollbar-color: #00d4ff #0a192f;
    }

    .review-container::-webkit-scrollbar {
        width: 8px;
    }

    .review-container::-webkit-scrollbar-track {
        background: #0a192f;
        border-radius: 10px;
    }

    .review-container::-webkit-scrollbar-thumb {
        background: #00d4ff;
        border-radius: 10px;
    }

    /* Review Styling */
    .review-box {
        background: rgba(144, 238, 144, 0.15);
        border-left: 4px solid #90ee90;
        padding: 15px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 12px;
    }

    /* Badge Styling */
    .badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }

    .badge-positive { background: rgba(144, 238, 144, 0.2); color: #90ee90; border: 1px solid rgba(144, 238, 144, 0.3); }
    .badge-neutral { background: rgba(255, 193, 7, 0.2); color: #ffc107; border: 1px solid rgba(255, 193, 7, 0.3); }
    .badge-negative { background: rgba(255, 99, 71, 0.2); color: #ff6347; border: 1px solid rgba(255, 99, 71, 0.3); }

    /* Animations */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .animate-fade-in { animation: fadeIn 0.6s ease-out; }

    /* Responsive Design */
    @media (max-width: 768px) {
        .stApp { padding: 10px; }
        section[data-testid="stSidebar"] { width: 100%; max-width: 100%; }
        .drug-card { padding: 15px; }
        h1 { font-size: 2rem; }
        .stButton>button { padding: 10px 16px; font-size: 14px; }
        .review-container { max-height: 200px; }
    }

    /* Accessibility */
    .stButton>button:focus { outline: 2px solid #00d4ff; outline-offset: 2px; }
    .stTextInput > div > div > input:focus, .stTextArea textarea:focus { outline: 2px solid #00d4ff; }
    .stSelectbox > div > div:focus { outline: 2px solid #00d4ff; }
    [role="radiogroup"] label:focus { outline: 2px solid #00d4ff; }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def display_rating(rating, max_rating=10):
    full_stars = int(rating / (max_rating/5))
    empty_stars = 5 - full_stars
    return "★" * full_stars + "☆" * empty_stars

def create_drug_card(drug_name, positive_reviews, avg_rating, total_reviews, rank=None):
    badge = '<span class="badge badge-positive">Highly Recommended</span>' if avg_rating >= 8 else \
            '<span class="badge badge-neutral">Recommended</span>' if avg_rating >= 6 else \
            '<span class="badge badge-negative">Mixed Reviews</span>'
    rank_badge = f'<span style="background:#00d4ff; color:#0a192f; padding:4px 12px; border-radius:20px; font-weight:700; margin-right:10px">#{rank}</span>' if rank else ''
    effectiveness = int((avg_rating / 10) * 100)
    return f"""
    <div class="drug-card animate-fade-in" role="region" aria-label="Drug recommendation card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px">
            <div>
                {rank_badge}<span style="font-size:20px; font-weight:600">{drug_name}</span> {badge}
            </div>
            <div style="color:#ffd700">{display_rating(avg_rating)} {avg_rating:.1f}/10</div>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr 2fr; gap:15px; margin:10px 0">
            <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.05); border-radius:10px">
                <div style="font-size:22px; font-weight:700; color:#90ee90">{int(positive_reviews)}</div>
                <div style="font-size:12px; opacity:0.7">Positive Reviews</div>
            </div>
            <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.05); border-radius:10px">
                <div style="font-size:22px; font-weight:700; color:#00d4ff">{int(total_reviews)}</div>
                <div style="font-size:12px; opacity:0.7">Total Reviews</div>
            </div>
            <div style="padding:10px; background:rgba(255,255,255,0.05); border-radius:10px">
                <div style="font-size:12px; opacity:0.7; margin-bottom:5px">Effectiveness</div>
                <div style="width:100%; background:rgba(255,255,255,0.1); height:8px; border-radius:4px; overflow:hidden">
                    <div style="width:{effectiveness}%; background:linear-gradient(90deg, #00d4ff, #90ee90); height:8px"></div>
                </div>
            </div>
        </div>
    </div>
    """

def predict_sentiment(review_input, condition_input, encoder, model):
    try:
        # Prepare the input data
        input_df = pd.DataFrame([{"review": review_input, "condition": condition_input.lower()}])
        
        # Ensure encoder is not None and transform the data
        if encoder is None:
            st.error("❌ Encoder not loaded.")
            return None
        
        # Encoding the input data to the correct format for the model
        transformed_input = encoder.transform(input_df)
        
        # Predicting the sentiment
        prediction = model.predict(transformed_input)
        
        # Return whether the prediction is positive (1) or negative (0)
        return prediction[0] == 1
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None


# Dark mode toggle function and session state initialization
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding:20px 0">
            <h2 style="margin:0; color:#00d4ff">MedGuide</h2>
            <p style="opacity:0.7; font-size:14px">Smart Drug Recommender</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.button("🌙 Toggle Dark Mode", on_click=toggle_dark_mode)
    
    st.markdown("### 🔍 Navigation", help="Select a page to explore MedGuide features")
    page = st.radio("", ["🏠 Home", "🧪 Predict Review", "📊 Analytics", "ℹ️ About"], key="nav", label_visibility="collapsed")
    
    st.markdown('<p style="font-weight:500; margin:15px 0 5px">Quick Search</p>', unsafe_allow_html=True)
    quick_search = st.text_input("", placeholder="Search conditions...", label_visibility="collapsed")
    
    st.markdown("""
        <div style="background:rgba(0,212,255,0.1); padding:15px; border-radius:10px; margin-top:20px; border:1px solid rgba(0,212,255,0.3)">
            <h4 style="margin:0; color:#00d4ff">👋 Welcome!</h4>
            <p style="margin:5px 0 0; font-size:14px; opacity:0.8">
                Discover medications based on real patient reviews.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="position:absolute; bottom:20px; left:20px; font-size:12px; opacity:0.7">
            v2.2.0 | Updated May 2025
        </div>
    """, unsafe_allow_html=True)

# Apply dark mode class
st.markdown(f'<div class="{"dark-mode" if st.session_state.dark_mode else ""}">', unsafe_allow_html=True)

# Header
logo_path = "medguide_logo.png"
col1, col2 = st.columns([1, 5])
if os.path.exists(logo_path):
    with col1:
        st.image(logo_path, width=100)
with col2:
    st.title("MedGuide - Smart Drug Recommender")

# Home Page
if page == "🏠 Home":
    st.markdown("""
        <div style="background:rgba(255,255,255,0.05); padding:30px; border-radius:15px; text-align:center; margin-bottom:30px">
            <h2 style="margin:0">Find Top Medications</h2>
            <p style="opacity:0.8; font-size:16px">AI-powered insights from patient reviews to guide your treatment decisions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Updated search bar to a dropdown with only the three specified categories
        condition_input = st.selectbox("Enter Medical Condition", 
                                      options=["Depression", "High Blood Pressure", "Diabetes, Type 2"],
                                      index=None,
                                      placeholder="Select a condition...",
                                      key="condition")
    with col2:
        search_btn = st.button("🔍 Search", use_container_width=True)
    
    if quick_search:
        if quick_search.lower() in ["depression", "high blood pressure", "diabetes, type 2"]:
            condition_input = quick_search.title()
            search_btn = True
        else:
            st.warning("⚠️ Quick search only supports: Depression, High Blood Pressure, Diabetes, Type 2")

    if condition_input or search_btn:
        with st.spinner("Analyzing reviews..."):
            if not condition_input:
                st.warning("⚠️ Please select a condition to search.")
            else:
                condition_filtered = data[data['condition'].str.lower() == condition_input.lower()]
                
                if condition_filtered.empty:
                    st.warning(f"⚠️ No reviews found for '{condition_input}'. Please try another condition.")
                else:
                    st.success(f"✅ Found {len(condition_filtered)} reviews for {condition_input}")
                    
                    condition_filtered["is_positive"] = condition_filtered["rating"] >= 7
                    drug_stats = (
                        condition_filtered.groupby('drugName')
                        .agg(avg_rating=('rating', 'mean'),
                             num_reviews=('rating', 'count'),
                             positive_reviews=('is_positive', 'sum'))
                        .sort_values(by=['positive_reviews', 'avg_rating'], ascending=False)
                        .reset_index()
                    )
                    
                    tab1, tab2 = st.tabs(["🏆 Top Picks", "📋 All Drugs"])
                    with tab1:
                        for i, row in drug_stats.head(3).iterrows():
                            st.markdown(create_drug_card(row['drugName'], row['positive_reviews'], row['avg_rating'], row['num_reviews'], i+1), unsafe_allow_html=True)
                            review_key = f"rev_{i}"
                            
                            if st.button(f"📝 Reviews for {row['drugName']}", key=review_key):
                                # Initialize session state for loading reviews
                                if f"reviews_visible_{review_key}" not in st.session_state:
                                    st.session_state[f"reviews_visible_{review_key}"] = True
                                    st.session_state[f"reviews_count_{review_key}"] = 3  # Initial number of reviews to show
                                else:
                                    st.session_state[f"reviews_visible_{review_key}"] = not st.session_state[f"reviews_visible_{review_key}"]
                            
                            if st.session_state.get(f"reviews_visible_{review_key}", False):
                                reviews = condition_filtered[condition_filtered['drugName'] == row['drugName']]
                                pos_reviews = reviews[reviews['rating'] >= 7][['review', 'rating']].sort_values(by='rating', ascending=False)
                                
                                if pos_reviews.empty:
                                    st.info("ℹ️ No positive reviews found for this drug.")
                                else:
                                    # Scrollable review section
                                    st.markdown('<div class="review-container" role="region" aria-label="Patient reviews">', unsafe_allow_html=True)
                                    for r in pos_reviews.head(st.session_state.get(f"reviews_count_{review_key}", 3)).itertuples():
                                        st.markdown(f"""
                                            <div class="review-box" role="article">
                                                <div style="color:#ffd700">{display_rating(r.rating)} {r.rating}/10</div>
                                                <p style="font-style:italic">"{r.review[:250]}{'...' if len(r.review) > 250 else ''}"</p>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Load More button
                                    if st.session_state.get(f"reviews_count_{review_key}", 3) < len(pos_reviews):
                                        if st.button("⬇ Load More Reviews", key=f"load_more_{review_key}"):
                                            st.session_state[f"reviews_count_{review_key}"] = st.session_state.get(f"reviews_count_{review_key}", 3) + 3
                                            st.experimental_rerun()
                    
                    with tab2:
                        for i, row in drug_stats.iloc[3:].iterrows():
                            st.markdown(create_drug_card(row['drugName'], row['positive_reviews'], row['avg_rating'], row['num_reviews']), unsafe_allow_html=True)

# Predict Review Page
# Predict Review Page
# Predict Review Page
elif page == "🧪 Predict Review":
    st.markdown(""" 
        <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; margin-bottom:30px">
            <h2>🧪 Analyze Review Sentiment</h2>
            <p style="opacity:0.8">Use our AI to predict whether a patient review is positive or negative.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Layout for input fields
    col1, col2 = st.columns([2, 1])
    with col1:
        # Collecting the patient review input
        review_input = st.text_area("Patient Review", placeholder="e.g. This medication reduced my symptoms...", height=150, key="review")
    with col2:
        # Collecting the condition input
        condition_input = st.selectbox("Medical Condition", 
                                      options=["Depression", "High Blood Pressure", "Diabetes, Type 2"],
                                      index=None,
                                      placeholder="Select a condition...",
                                      key="condition_pred")
        # Analyze button to trigger the sentiment prediction
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    
    # Display example reviews
    with st.expander("💡 Example Reviews"):
        st.markdown(""" 
            **Positive**: "This drug helped my anxiety significantly with only mild side effects."  
            **Negative**: "Experienced severe nausea and no improvement in symptoms."
        """)
    
    # When the analyze button is clicked and inputs are provided
    if analyze_btn and review_input and condition_input:
        with st.spinner("Analyzing sentiment..."):
            # Predict the sentiment based on the review and condition
            is_positive = predict_sentiment(review_input, condition_input, encoder, model)
            
            if is_positive is not None:
                # Display sentiment result based on prediction
                sentiment_color = "#90ee90" if is_positive else "#ff6347"
                sentiment_icon = "✅" if is_positive else "❌"
                sentiment_text = "Positive" if is_positive else "Negative"
                sentiment_desc = "This review expresses satisfaction with the medication." if is_positive else \
                                "This review indicates dissatisfaction with the medication."
                
                # Display the prediction outcome
                st.markdown(f"""
                    <div class="animate-fade-in" style="background:rgba(255,255,255,0.1); backdrop-filter:blur(10px); border-radius:20px; padding:30px; text-align:center; margin:20px 0">
                        <div style="font-size:60px; margin-bottom:10px">{sentiment_icon}</div>
                        <h2 style="color:{sentiment_color}">{sentiment_text} Sentiment</h2>
                        <p>{sentiment_desc}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics like review length, estimated rating, etc.
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Review Length", f"{len(review_input)} chars")
                with col2:
                    st.metric("Estimated Rating", f"{8.5 if is_positive else 3.5}/10")
                with col3:
                    st.metric("Word Count", len(review_input.split()))
            else:
                st.warning("⚠️ Unable to predict sentiment. Please check your input.")



# Analytics Page
elif page == "📊 Analytics":
    st.markdown("""
        <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; margin-bottom:30px">
            <h2>📊 Medication Insights</h2>
            <p style="opacity:0.8">Explore trends, ratings, and sentiment across medications and conditions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔍 Compare", "💬 Sentiment"])
    
    with tab1:
        if not data.empty:
            col1, col2 = st.columns(2)
            with col1:
                top_conditions = data['condition'].value_counts().head(5).reset_index()
                top_conditions.columns = ['Condition', 'Reviews']
                fig = px.bar(top_conditions, x='Condition', y='Reviews', title="Top Conditions by Reviews",
                             color='Reviews', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                top_drugs = data.groupby('drugName')['rating'].agg(['mean', 'count']).sort_values('count', ascending=False).head(10).reset_index()
                top_drugs.columns = ['Drug', 'Avg Rating', 'Reviews']
                fig = px.bar(top_drugs, x='Drug', y='Avg Rating', title="Top Drugs by Rating",
                             color='Reviews', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Rating", f"{data['rating'].mean():.1f}/10")
            with col2:
                st.metric("Total Reviews", f"{len(data):,}")
            with col3:
                st.metric("Positive Rate", f"{(data['rating'] >= 7).mean() * 100:.1f}%")
            with col4:
                st.metric("Medications", f"{data['drugName'].nunique():,}")
    
    with tab2:
        if not data.empty:
            condition = st.selectbox("Select Condition", sorted(data['condition'].unique()))
            condition_data = data[data['condition'] == condition]
            if not condition_data.empty:
                drug_comparison = condition_data.groupby('drugName')['rating'].agg(['mean', 'count']).reset_index()
                drug_comparison = drug_comparison[drug_comparison['count'] >= 5].sort_values('mean', ascending=False)
                fig = px.bar(drug_comparison, x='drugName', y='mean', title=f"Drug Ratings for {condition}",
                             color='count', color_continuous_scale='Teal')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if not data.empty:
            data['sentiment'] = data['rating'].apply(lambda x: "Positive" if x >= 7 else "Negative")
            sentiment_counts = data['sentiment'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution",
                         color_discrete_sequence=['#90ee90', '#ff6347'])
            st.plotly_chart(fig, use_container_width=True)

# About Page
elif page == "ℹ️ About":
    st.markdown("""
        <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; margin-bottom:30px">
            <h2>ℹ️ About MedGuide</h2>
            <p style="opacity:0.8">Empowering healthcare decisions with AI-driven insights from patient reviews.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
            <div style="background:rgba(255,255,255,0.05); padding:25px; border-radius:15px; margin-bottom:20px">
                <h3>Our Mission</h3>
                <p>MedGuide bridges the gap between medical data and patient care by providing reliable, data-driven medication recommendations.</p>
            </div>
            <div style="background:rgba(255,255,255,0.05); padding:25px; border-radius:15px">
                <h3>Features</h3>
                <ul>
                    <li>🔍 Smart medication search</li>
                    <li>🧠 AI sentiment analysis</li>
                    <li>📊 Interactive analytics</li>
                    <li>🔒 Secure and private</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background:rgba(255,255,255,0.05); padding:25px; border-radius:15px; text-align:center">
                <h3>Meet the Team</h3>
                <div style="width:100px; height:100px; border-radius:50%; background:linear-gradient(135deg,#00d4ff,#90ee90); margin:0 auto 15px; display:flex; align-items:center; justify-content:center; font-size:40px">👨‍💻</div>
                <h4>Shaikh Akbar Ali</h4>
                <p style="opacity:0.8">Lead Developer & Data Scientist</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align:center; padding:20px; margin-top:40px; border-top:1px solid rgba(255,255,255,0.1); font-size:14px; opacity:0.8">
        MedGuide v2.2.0 | Made with 💙 by Shaikh Akbar Ali | © 2025
    </div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


'''
'''
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

# ===================== PAGE CONFIGURATION =====================
st.set_page_config(
    page_title="MedGuide - Drug Recommender",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ===================== CUSTOM CSS FOR UI =====================
st.markdown("""
    <style>
    /* [CSS omitted for brevity: use original CSS block from previous code] */
    </style>
""", unsafe_allow_html=True)

# ===================== HELPER FUNCTIONS =====================
def display_rating(rating, max_rating=10):
    full_stars = int(rating / (max_rating/5))
    empty_stars = 5 - full_stars
    return "★" * full_stars + "☆" * empty_stars

def create_drug_card(drug_name, positive_reviews, avg_rating, total_reviews, rank=None):
    badge = '<span class="badge badge-positive">Highly Recommended</span>' if avg_rating >= 8 else \
            '<span class="badge badge-neutral">Recommended</span>' if avg_rating >= 6 else \
            '<span class="badge badge-negative">Mixed Reviews</span>'
    rank_badge = f'<span style="background:#00d4ff; color:#0a192f; padding:4px 12px; border-radius:20px; font-weight:700; margin-right:10px">#{rank}</span>' if rank else ''
    effectiveness = int((avg_rating / 10) * 100)
    return f"""
    <div class="drug-card animate-fade-in" role="region" aria-label="Drug recommendation card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px">
            <div>
                {rank_badge}<span style="font-size:20px; font-weight:600">{drug_name}</span> {badge}
            </div>
            <div style="color:#ffd700">{display_rating(avg_rating)} {avg_rating:.1f}/10</div>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr 2fr; gap:15px; margin:10px 0">
            <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.05); border-radius:10px">
                <div style="font-size:22px; font-weight:700; color:#90ee90">{int(positive_reviews)}</div>
                <div style="font-size:12px; opacity:0.7">Positive Reviews</div>
            </div>
            <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.05); border-radius:10px">
                <div style="font-size:22px; font-weight:700; color:#00d4ff">{int(total_reviews)}</div>
                <div style="font-size:12px; opacity:0.7">Total Reviews</div>
            </div>
            <div style="padding:10px; background:rgba(255,255,255,0.05); border-radius:10px">
                <div style="font-size:12px; opacity:0.7; margin-bottom:5px">Effectiveness</div>
                <div style="width:100%; background:rgba(255,255,255,0.1); height:8px; border-radius:4px; overflow:hidden">
                    <div style="width:{effectiveness}%; background:linear-gradient(90deg, #00d4ff, #90ee90); height:8px"></div>
                </div>
            </div>
        </div>
    </div>
    """

def predict_sentiment(review_input, condition_input, encoder, model):
    try:
        input_df = pd.DataFrame([{"review": review_input, "condition": condition_input.lower()}])
        if encoder is None:
            st.error("❌ Encoder not loaded.")
            return None
        transformed_input = encoder.transform(input_df)
        prediction = model.predict(transformed_input)
        return prediction[0] == 1
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None

# --- AI Review Summarizer ---
def summarize_review(text, api_key=None):
    if not api_key:
        st.warning("No API key provided for summarization.")
        return ""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "inputs": f"summarize: {text}",
            "parameters": {"max_length": 60, "min_length": 15, "do_sample": False}
        }
        response = requests.post(
            "https://api-inference.huggingface.co/models/t5-small",
            headers=headers,
            json=payload,
            timeout=10
        )
        if response.ok:
            summary = response.json()[0]['summary_text']
            return summary
        else:
            st.error("Summarization API error.")
            return ""
    except Exception as e:
        st.error(f"Summarization failed: {e}")
        return ""

# --- LLM Chatbot Assistant ---
def medguide_chat(query, api_key=None):
    if not api_key:
        st.warning("No API key provided for chat.")
        return ""
    try:
        import openai
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert medical assistant. Only answer with factual, non-diagnostic information."},
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Chat failed: {e}")
        return ""

# --- Medicine App Features (PharmEasy-style) ---
def get_medicine_suggestions(query, med_list):
    return [m for m in med_list if query.lower() in m.lower()][:5]

def get_fake_medicine_db():
    return [
        {"name": "Paracetamol", "uses": "Fever, Pain", "manufacturer": "Acme Pharma", "price": 15, "salt": "Paracetamol", "alternatives": ["Crocin", "Dolo"]},
        {"name": "Amoxicillin", "uses": "Bacterial Infection", "manufacturer": "BioMed", "price": 95, "salt": "Amoxicillin", "alternatives": ["Mox", "Amoxil"]},
        {"name": "Atorvastatin", "uses": "Cholesterol", "manufacturer": "StatCare", "price": 120, "salt": "Atorvastatin", "alternatives": ["Lipitor", "Tonact"]},
        {"name": "Metformin", "uses": "Diabetes", "manufacturer": "GlucoPharma", "price": 80, "salt": "Metformin", "alternatives": ["Gluformin", "Obimet"]},
    ]

def fake_drug_interactions(med1, med2):
    if set([med1, med2]) == set(["Metformin", "Atorvastatin"]):
        return True, "Metformin and Atorvastatin may interact and increase risk of muscle pain."
    return False, ""

def render_map(city=""):
    st.image("https://i.imgur.com/6zK6wQk.png", caption=f"Nearby pharmacies in {city or 'your area'} (Demo)")

# ===================== DARK MODE =====================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# ===================== SIDEBAR NAVIGATION =====================
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding:20px 0">
            <h2 style="margin:0; color:#00d4ff">MedGuide</h2>
            <p style="opacity:0.7; font-size:14px">Smart Drug Recommender</p>
        </div>
    """, unsafe_allow_html=True)
    st.button("🌙 Toggle Dark Mode", on_click=toggle_dark_mode)
    st.markdown("### 🔍 Navigation", help="Select a page to explore MedGuide features")
    page = st.radio(
        "",
        ["🏠 Home", "🧪 Predict Review", "📊 Analytics", "ℹ️ About", "💊 Pharmacy Tools"],
        key="nav",
        label_visibility="collapsed"
    )
    st.markdown('<p style="font-weight:500; margin:15px 0 5px">Quick Search</p>', unsafe_allow_html=True)
    quick_search = st.text_input("", placeholder="Search conditions...", label_visibility="collapsed")
    st.markdown("""
        <div style="background:rgba(0,212,255,0.1); padding:15px; border-radius:10px; margin-top:20px; border:1px solid rgba(0,212,255,0.3)">
            <h4 style="margin:0; color:#00d4ff">👋 Welcome!</h4>
            <p style="margin:5px 0 0; font-size:14px; opacity:0.8">
                Discover medications based on real patient reviews.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style="position:absolute; bottom:20px; left:20px; font-size:12px; opacity:0.7">
            v2.2.0 | Updated May 2025
        </div>
    """, unsafe_allow_html=True)
    st.markdown("### 🤖 AI Tools")
    ai_tool = st.radio("Select AI Tool", ["None", "Review Summarizer", "MedGuide Chatbot"], key="ai_tools")

api_key = st.secrets.get("HF_API_KEY") or st.secrets.get("OPENAI_API_KEY") or st.text_input(
    "🔑 Enter your HuggingFace or OpenAI API key:", type="password", key="api_key_input"
)

# ===================== DARK MODE WRAPPER =====================
st.markdown(f'<div class="{"dark-mode" if st.session_state.dark_mode else ""}">', unsafe_allow_html=True)

# ===================== AI TOOLS SECTION =====================
if ai_tool == "Review Summarizer":
    st.header("📝 AI Review Summarizer")
    review_text = st.text_area("Paste patient review to summarize", height=180)
    if st.button("Summarize", type="primary"):
        with st.spinner("Summarizing..."):
            summary = summarize_review(review_text, api_key)
            if summary:
                st.success("Summary:")
                st.write(summary)

elif ai_tool == "MedGuide Chatbot":
    st.header("💬 MedGuide AI Chat Assistant")
    chat_query = st.text_input("Ask any drug or condition question (no personal data):")
    if st.button("Ask", type="primary"):
        with st.spinner("Thinking..."):
            answer = medguide_chat(chat_query, api_key)
            if answer:
                st.success("AI Answer:")
                st.write(answer)

# ===================== HEADER =====================
logo_path = "medguide_logo.png"
col1, col2 = st.columns([1, 5])
if os.path.exists(logo_path):
    with col1:
        st.image(logo_path, width=100)
with col2:
    st.title("MedGuide - Smart Drug Recommender")

# ===================== MAIN PAGES =====================
if page == "🏠 Home":
    st.markdown("""
        <div style="background:rgba(255,255,255,0.05); padding:30px; border-radius:15px; text-align:center; margin-bottom:30px">
            <h2 style="margin:0">Find Top Medications</h2>
            <p style="opacity:0.8; font-size:16px">AI-powered insights from patient reviews to guide your treatment decisions.</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        condition_input = st.selectbox("Enter Medical Condition", 
                                      options=["Depression", "High Blood Pressure", "Diabetes, Type 2"],
                                      index=None,
                                      placeholder="Select a condition...",
                                      key="condition")
    with col2:
        search_btn = st.button("🔍 Search", use_container_width=True)
    if quick_search:
        if quick_search.lower() in ["depression", "high blood pressure", "diabetes, type 2"]:
            condition_input = quick_search.title()
            search_btn = True
        else:
            st.warning("⚠️ Quick search only supports: Depression, High Blood Pressure, Diabetes, Type 2")
    if condition_input or search_btn:
        with st.spinner("Analyzing reviews..."):
            if not condition_input:
                st.warning("⚠️ Please select a condition to search.")
            else:
                condition_filtered = data[data['condition'].str.lower() == condition_input.lower()]
                if condition_filtered.empty:
                    st.warning(f"⚠️ No reviews found for '{condition_input}'. Please try another condition.")
                else:
                    st.success(f"✅ Found {len(condition_filtered)} reviews for {condition_input}")
                    condition_filtered["is_positive"] = condition_filtered["rating"] >= 7
                    drug_stats = (
                        condition_filtered.groupby('drugName')
                        .agg(avg_rating=('rating', 'mean'),
                             num_reviews=('rating', 'count'),
                             positive_reviews=('is_positive', 'sum'))
                        .sort_values(by=['positive_reviews', 'avg_rating'], ascending=False)
                        .reset_index()
                    )
                    tab1, tab2 = st.tabs(["🏆 Top Picks", "📋 All Drugs"])
                    with tab1:
                        for i, row in drug_stats.head(3).iterrows():
                            st.markdown(create_drug_card(row['drugName'], row['positive_reviews'], row['avg_rating'], row['num_reviews'], i+1), unsafe_allow_html=True)
                            review_key = f"rev_{i}"
                            if st.button(f"📝 Reviews for {row['drugName']}", key=review_key):
                                if f"reviews_visible_{review_key}" not in st.session_state:
                                    st.session_state[f"reviews_visible_{review_key}"] = True
                                    st.session_state[f"reviews_count_{review_key}"] = 3
                                else:
                                    st.session_state[f"reviews_visible_{review_key}"] = not st.session_state[f"reviews_visible_{review_key}"]
                            if st.session_state.get(f"reviews_visible_{review_key}", False):
                                reviews = condition_filtered[condition_filtered['drugName'] == row['drugName']]
                                pos_reviews = reviews[reviews['rating'] >= 7][['review', 'rating']].sort_values(by='rating', ascending=False)
                                if pos_reviews.empty:
                                    st.info("ℹ️ No positive reviews found for this drug.")
                                else:
                                    st.markdown('<div class="review-container" role="region" aria-label="Patient reviews">', unsafe_allow_html=True)
                                    for r in pos_reviews.head(st.session_state.get(f"reviews_count_{review_key}", 3)).itertuples():
                                        st.markdown(f"""
                                            <div class="review-box" role="article">
                                                <div style="color:#ffd700">{display_rating(r.rating)} {r.rating}/10</div>
                                                <p style="font-style:italic">"{r.review[:250]}{'...' if len(r.review) > 250 else ''}"</p>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    if st.session_state.get(f"reviews_count_{review_key}", 3) < len(pos_reviews):
                                        if st.button("⬇ Load More Reviews", key=f"load_more_{review_key}"):
                                            st.session_state[f"reviews_count_{review_key}"] = st.session_state.get(f"reviews_count_{review_key}", 3) + 3
                                            st.experimental_rerun()
                    with tab2:
                        for i, row in drug_stats.iloc[3:].iterrows():
                            st.markdown(create_drug_card(row['drugName'], row['positive_reviews'], row['avg_rating'], row['num_reviews']), unsafe_allow_html=True)

elif page == "🧪 Predict Review":
    st.markdown(""" 
        <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; margin-bottom:30px">
            <h2>🧪 Analyze Review Sentiment</h2>
            <p style="opacity:0.8">Use our AI to predict whether a patient review is positive or negative.</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        review_input = st.text_area("Patient Review", placeholder="e.g. This medication reduced my symptoms...", height=150, key="review")
    with col2:
        condition_input = st.selectbox("Medical Condition", 
                                      options=["Depression", "High Blood Pressure", "Diabetes, Type 2"],
                                      index=None,
                                      placeholder="Select a condition...",
                                      key="condition_pred")
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    with st.expander("💡 Example Reviews"):
        st.markdown(""" 
            **Positive**: "This drug helped my anxiety significantly with only mild side effects."  
            **Negative**: "Experienced severe nausea and no improvement in symptoms."
        """)
    if analyze_btn and review_input and condition_input:
        with st.spinner("Analyzing sentiment..."):
            is_positive = predict_sentiment(review_input, condition_input, encoder, model)
            if is_positive is not None:
                sentiment_color = "#90ee90" if is_positive else "#ff6347"
                sentiment_icon = "✅" if is_positive else "❌"
                sentiment_text = "Positive" if is_positive else "Negative"
                sentiment_desc = "This review expresses satisfaction with the medication." if is_positive else \
                                "This review indicates dissatisfaction with the medication."
                st.markdown(f"""
                    <div class="animate-fade-in" style="background:rgba(255,255,255,0.1); backdrop-filter:blur(10px); border-radius:20px; padding:30px; text-align:center; margin:20px 0">
                        <div style="font-size:60px; margin-bottom:10px">{sentiment_icon}</div>
                        <h2 style="color:{sentiment_color}">{sentiment_text} Sentiment</h2>
                        <p>{sentiment_desc}</p>
                    </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Review Length", f"{len(review_input)} chars")
                with col2:
                    st.metric("Estimated Rating", f"{8.5 if is_positive else 3.5}/10")
                with col3:
                    st.metric("Word Count", len(review_input.split()))
            else:
                st.warning("⚠️ Unable to predict sentiment. Please check your input.")

elif page == "📊 Analytics":
    st.markdown("""
        <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; margin-bottom:30px">
            <h2>📊 Medication Insights</h2>
            <p style="opacity:0.8">Explore trends, ratings, and sentiment across medications and conditions.</p>
        </div>
    """, unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔍 Compare", "💬 Sentiment"])
    with tab1:
        if not data.empty:
            col1, col2 = st.columns(2)
            with col1:
                top_conditions = data['condition'].value_counts().head(5).reset_index()
                top_conditions.columns = ['Condition', 'Reviews']
                fig = px.bar(top_conditions, x='Condition', y='Reviews', title="Top Conditions by Reviews",
                             color='Reviews', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                top_drugs = data.groupby('drugName')['rating'].agg(['mean', 'count']).sort_values('count', ascending=False).head(10).reset_index()
                top_drugs.columns = ['Drug', 'Avg Rating', 'Reviews']
                fig = px.bar(top_drugs, x='Drug', y='Avg Rating', title="Top Drugs by Rating",
                             color='Reviews', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Rating", f"{data['rating'].mean():.1f}/10")
            with col2:
                st.metric("Total Reviews", f"{len(data):,}")
            with col3:
                st.metric("Positive Rate", f"{(data['rating'] >= 7).mean() * 100:.1f}%")
            with col4:
                st.metric("Medications", f"{data['drugName'].nunique():,}")
    with tab2:
        if not data.empty:
            condition = st.selectbox("Select Condition", sorted(data['condition'].unique()))
            condition_data = data[data['condition'] == condition]
            if not condition_data.empty:
                drug_comparison = condition_data.groupby('drugName')['rating'].agg(['mean', 'count']).reset_index()
                drug_comparison = drug_comparison[drug_comparison['count'] >= 5].sort_values('mean', ascending=False)
                fig = px.bar(drug_comparison, x='drugName', y='mean', title=f"Drug Ratings for {condition}",
                             color='count', color_continuous_scale='Teal')
                st.plotly_chart(fig, use_container_width=True)
    with tab3:
        if not data.empty:
            data['sentiment'] = data['rating'].apply(lambda x: "Positive" if x >= 7 else "Negative")
            sentiment_counts = data['sentiment'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution",
                         color_discrete_sequence=['#90ee90', '#ff6347'])
            st.plotly_chart(fig, use_container_width=True)

elif page == "ℹ️ About":
    st.markdown("""
        <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; margin-bottom:30px">
            <h2>ℹ️ About MedGuide</h2>
            <p style="opacity:0.8">Empowering healthcare decisions with AI-driven insights from patient reviews.</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
            <div style="background:rgba(255,255,255,0.05); padding:25px; border-radius:15px; margin-bottom:20px">
                <h3>Our Mission</h3>
                <p>MedGuide bridges the gap between medical data and patient care by providing reliable, data-driven medication recommendations.</p>
            </div>
            <div style="background:rgba(255,255,255,0.05); padding:25px; border-radius:15px">
                <h3>Features</h3>
                <ul>
                    <li>🔍 Smart medication search</li>
                    <li>🧠 AI sentiment analysis</li>
                    <li>📊 Interactive analytics</li>
                    <li>🔒 Secure and private</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="background:rgba(255,255,255,0.05); padding:25px; border-radius:15px; text-align:center">
                <h3>Meet the Team</h3>
                <div style="width:100px; height:100px; border-radius:50%; background:linear-gradient(135deg,#00d4ff,#90ee90); margin:0 auto 15px; display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:48px; color:white; font-weight:bold;">SA</span>
                </div>
                <h4>Shaikh Akbar Ali</h4>
                <p style="opacity:0.8">Lead Developer & Data Scientist</p>
            </div>
        """, unsafe_allow_html=True)

elif page == "💊 Pharmacy Tools":
    st.header("💊 Pharmacy App Features (PharmEasy Style)")
    pharma_page = st.radio(
        "Choose Tool",
        [
            "Medicine Search", "Drug Alternatives",
            "Order (Upload Prescription)", "Pill Reminder",
            "Nearby Pharmacies", "Drug Interaction Checker"
        ],
        key="pharma_page"
    )
    med_db = get_fake_medicine_db()
    med_names = [m["name"] for m in med_db]
    if pharma_page == "Medicine Search":
        st.subheader("🔎 Medicine Search")
        query = st.text_input("Enter medicine name or salt:", key="pharma_search")
        if query:
            suggestions = get_medicine_suggestions(query, med_names)
            for s in suggestions:
                med = next((m for m in med_db if m["name"] == s), None)
                if med:
                    st.markdown(f"""
                    **{med['name']}**  
                    Uses: {med['uses']}  
                    Manufacturer: {med['manufacturer']}  
                    Price: ₹{med['price']} per strip  
                    Salt: {med['salt']}  
                    """)
                    st.markdown(f"*Alternatives*: {', '.join(med['alternatives'])}")
    elif pharma_page == "Drug Alternatives":
        st.subheader("💡 Drug Alternatives Finder")
        med = st.selectbox("Select a medicine", med_names)
        if med:
            selected = next((m for m in med_db if m["name"] == med), None)
            if selected:
                st.subheader(f"Alternatives to {med}:")
                for alt in selected["alternatives"]:
                    st.markdown(f"- {alt}")
    elif pharma_page == "Order (Upload Prescription)":
        st.subheader("📝 Upload Prescription (Demo)")
        uploaded = st.file_uploader("Upload your prescription image (jpg/png/pdf)")
        if uploaded:
            st.success("Prescription received! Our pharmacist will contact you soon (Demo).")
            st.image(uploaded, width=200)
    elif pharma_page == "Pill Reminder":
        st.subheader("⏰ Pill Reminder (Demo)")
        if "reminders" not in st.session_state:
            st.session_state.reminders = []
        med_name = st.text_input("Medicine Name:", key="reminder_med")
        time = st.time_input("Time:", key="reminder_time")
        freq = st.selectbox("Frequency", ["Once daily", "Morning & Night", "Every 8 hours"])
        if st.button("Add Reminder"):
            st.session_state.reminders.append({"med": med_name, "time": str(time), "freq": freq})
        st.subheader("Your Reminders")
        for r in st.session_state.reminders:
            st.info(f"{r['med']} at {r['time']} ({r['freq']})")
    elif pharma_page == "Nearby Pharmacies":
        st.subheader("🏪 Nearby Pharmacies (Demo)")
        city = st.text_input("Enter your city:", key="pharma_city")
        if city or st.button("Show Pharmacies"):
            render_map(city or "your area")
            st.markdown("**Note:** Map is for demo only. Real location services not enabled.")
    elif pharma_page == "Drug Interaction Checker":
        st.subheader("⚠️ Drug Interaction Checker (Demo)")
        med1 = st.selectbox("Medicine 1", med_names, key="int1")
        med2 = st.selectbox("Medicine 2", med_names, key="int2")
        if st.button("Check Interaction"):
            is_interact, msg = fake_drug_interactions(med1, med2)
            if is_interact:
                st.error(f"Warning: {msg}")
            else:
                st.success("✅ No significant interaction found in demo database.")

# ===================== FOOTER =====================
st.markdown("""
    <div style="text-align:center; padding:20px; margin-top:40px; border-top:1px solid rgba(255,255,255,0.1); font-size:14px; opacity:0.8">
        MedGuide v2.2.0 | Made with 💙 by Shaikh Akbar Ali | © 2025
    </div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
'''

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

# ===================== PAGE CONFIGURATION =====================
st.set_page_config(
    page_title="MedGuide - Drug Recommender",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ===================== USER DATA UPLOAD & AI LEARNING =====================
if "user_learned_data" not in st.session_state:
    st.session_state["user_learned_data"] = pd.DataFrame(columns=["drugName", "condition", "review", "rating"])

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📥 User Data Upload & AI Learning")
    up_file = st.file_uploader("Upload CSV/Excel with: drugName, condition, review, rating", type=["csv", "xlsx", "xls"])
    if up_file:
        try:
            if up_file.name.endswith(".csv"):
                df_new = pd.read_csv(up_file)
            else:
                df_new = pd.read_excel(up_file)
            cols = ["drugName", "condition", "review", "rating"]
            df_new = df_new[[c for c in cols if c in df_new.columns]]
            st.session_state["user_learned_data"] = pd.concat([st.session_state["user_learned_data"], df_new], ignore_index=True)
            st.sidebar.success(f"Added {len(df_new)} records. Total: {len(st.session_state['user_learned_data'])}")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

# Combine built-in + user-learned data for analytics and AI
combined_data = pd.concat([data, st.session_state["user_learned_data"]], ignore_index=True).copy()

# ===================== AUTO-LEARN SIMPLE SENTIMENT MODEL =====================
if len(st.session_state["user_learned_data"]) >= 6:
    combined_data["is_positive"] = combined_data["rating"].astype(float) >= 7
    X_train, X_test, y_train, y_test = train_test_split(
        combined_data["review"].astype(str), combined_data["is_positive"], test_size=0.2, random_state=42, stratify=combined_data["is_positive"]
    )
    sentiment_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000)),
        ("clf", LogisticRegression(solver="liblinear"))
    ])
    sentiment_pipe.fit(X_train, y_train)
    st.session_state["user_sentiment_model"] = sentiment_pipe
    st.session_state["user_sentiment_acc"] = sentiment_pipe.score(X_test, y_test)
    st.sidebar.success(f"AI learned! User model accuracy: {st.session_state['user_sentiment_acc']:.2f}")

# ===================== CUSTOM CSS FOR UI =====================
st.markdown("""
    <style>
    /* [CSS omitted for brevity: use your original CSS block here] */
    </style>
""", unsafe_allow_html=True)

# ===================== HELPER FUNCTIONS =====================
def display_rating(rating, max_rating=10):
    full_stars = int(rating / (max_rating/5))
    empty_stars = 5 - full_stars
    return "★" * full_stars + "☆" * empty_stars

def create_drug_card(drug_name, positive_reviews, avg_rating, total_reviews, rank=None):
    badge = '<span class="badge badge-positive">Highly Recommended</span>' if avg_rating >= 8 else \
            '<span class="badge badge-neutral">Recommended</span>' if avg_rating >= 6 else \
            '<span class="badge badge-negative">Mixed Reviews</span>'
    rank_badge = f'<span style="background:#00d4ff; color:#0a192f; padding:4px 12px; border-radius:20px; font-weight:700; margin-right:10px">#{rank}</span>' if rank else ''
    effectiveness = int((avg_rating / 10) * 100)
    return f"""
    <div class="drug-card animate-fade-in" role="region" aria-label="Drug recommendation card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px">
            <div>
                {rank_badge}<span style="font-size:20px; font-weight:600">{drug_name}</span> {badge}
            </div>
            <div style="color:#ffd700">{display_rating(avg_rating)} {avg_rating:.1f}/10</div>
        </div>
        <div style="display:grid; grid-template-columns: 1fr 1fr 2fr; gap:15px; margin:10px 0">
            <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.05); border-radius:10px">
                <div style="font-size:22px; font-weight:700; color:#90ee90">{int(positive_reviews)}</div>
                <div style="font-size:12px; opacity:0.7">Positive Reviews</div>
            </div>
            <div style="text-align:center; padding:10px; background:rgba(255,255,255,0.05); border-radius:10px">
                <div style="font-size:22px; font-weight:700; color:#00d4ff">{int(total_reviews)}</div>
                <div style="font-size:12px; opacity:0.7">Total Reviews</div>
            </div>
            <div style="padding:10px; background:rgba(255,255,255,0.05); border-radius:10px">
                <div style="font-size:12px; opacity:0.7; margin-bottom:5px">Effectiveness</div>
                <div style="width:100%; background:rgba(255,255,255,0.1); height:8px; border-radius:4px; overflow:hidden">
                    <div style="width:{effectiveness}%; background:linear-gradient(90deg, #00d4ff, #90ee90); height:8px"></div>
                </div>
            </div>
        </div>
    </div>
    """

def predict_sentiment(review_input, condition_input, encoder, model):
    try:
        input_df = pd.DataFrame([{"review": review_input, "condition": condition_input.lower()}])
        if encoder is None:
            st.error("❌ Encoder not loaded.")
            return None
        transformed_input = encoder.transform(input_df)
        prediction = model.predict(transformed_input)
        return prediction[0] == 1
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None

# --- AI Review Summarizer ---
def summarize_review(text, api_key=None):
    if not api_key:
        st.warning("No API key provided for summarization.")
        return ""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "inputs": f"summarize: {text}",
            "parameters": {"max_length": 60, "min_length": 15, "do_sample": False}
        }
        response = requests.post(
            "https://api-inference.huggingface.co/models/t5-small",
            headers=headers,
            json=payload,
            timeout=10
        )
        if response.ok:
            summary = response.json()[0]['summary_text']
            return summary
        else:
            st.error("Summarization API error.")
            return ""
    except Exception as e:
        st.error(f"Summarization failed: {e}")
        return ""

# --- LLM Chatbot Assistant ---
def medguide_chat(query, api_key=None):
    if not api_key:
        st.warning("No API key provided for chat.")
        return ""
    try:
        import openai
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert medical assistant. Only answer with factual, non-diagnostic information."},
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Chat failed: {e}")
        return ""

# --- Medicine App Features (PharmEasy-style) ---
def get_medicine_suggestions(query, med_list):
    return [m for m in med_list if query.lower() in m.lower()][:5]

def get_fake_medicine_db():
    return [
        {"name": "Paracetamol", "uses": "Fever, Pain", "manufacturer": "Acme Pharma", "price": 15, "salt": "Paracetamol", "alternatives": ["Crocin", "Dolo"]},
        {"name": "Amoxicillin", "uses": "Bacterial Infection", "manufacturer": "BioMed", "price": 95, "salt": "Amoxicillin", "alternatives": ["Mox", "Amoxil"]},
        {"name": "Atorvastatin", "uses": "Cholesterol", "manufacturer": "StatCare", "price": 120, "salt": "Atorvastatin", "alternatives": ["Lipitor", "Tonact"]},
        {"name": "Metformin", "uses": "Diabetes", "manufacturer": "GlucoPharma", "price": 80, "salt": "Metformin", "alternatives": ["Gluformin", "Obimet"]},
    ]

def fake_drug_interactions(med1, med2):
    if set([med1, med2]) == set(["Metformin", "Atorvastatin"]):
        return True, "Metformin and Atorvastatin may interact and increase risk of muscle pain."
    return False, ""

def render_map(city=""):
    st.image("https://i.imgur.com/6zK6wQk.png", caption=f"Nearby pharmacies in {city or 'your area'} (Demo)")

# DARK MODE
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# SIDEBAR - already present above, keep as is...

# ===================== DARK MODE WRAPPER =====================
st.markdown(f'<div class="{"dark-mode" if st.session_state.dark_mode else ""}">', unsafe_allow_html=True)

# ===================== AI TOOLS SECTION - as in your code =====================
# ... (unchanged) ...

# ===================== HEADER - as in your code =====================
# ... (unchanged) ...

# ===================== MAIN PAGES - MODIFIED TO USE combined_data =====================
if page == "🏠 Home":
    st.markdown("""
        <div style="background:rgba(255,255,255,0.05); padding:30px; border-radius:15px; text-align:center; margin-bottom:30px">
            <h2 style="margin:0">Find Top Medications</h2>
            <p style="opacity:0.8; font-size:16px">AI-powered insights from patient reviews to guide your treatment decisions.</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        condition_input = st.selectbox("Enter Medical Condition", 
                                      options=sorted(combined_data['condition'].dropna().unique()),
                                      index=None,
                                      placeholder="Select a condition...",
                                      key="condition")
    with col2:
        search_btn = st.button("🔍 Search", use_container_width=True)
    if quick_search:
        if quick_search.lower() in [c.lower() for c in combined_data['condition'].unique()]:
            condition_input = quick_search.title()
            search_btn = True
        else:
            st.warning(f"⚠️ Quick search only supports: {', '.join(sorted(set(combined_data['condition'].unique())))}")
    if condition_input or search_btn:
        with st.spinner("Analyzing reviews..."):
            if not condition_input:
                st.warning("⚠️ Please select a condition to search.")
            else:
                condition_filtered = combined_data[combined_data['condition'].str.lower() == condition_input.lower()]
                if condition_filtered.empty:
                    st.warning(f"⚠️ No reviews found for '{condition_input}'. Please try another condition.")
                else:
                    st.success(f"✅ Found {len(condition_filtered)} reviews for {condition_input}")
                    condition_filtered["is_positive"] = condition_filtered["rating"] >= 7
                    drug_stats = (
                        condition_filtered.groupby('drugName')
                        .agg(avg_rating=('rating', 'mean'),
                             num_reviews=('rating', 'count'),
                             positive_reviews=('is_positive', 'sum'))
                        .sort_values(by=['positive_reviews', 'avg_rating'], ascending=False)
                        .reset_index()
                    )
                    tab1, tab2 = st.tabs(["🏆 Top Picks", "📋 All Drugs"])
                    with tab1:
                        for i, row in drug_stats.head(3).iterrows():
                            st.markdown(create_drug_card(row['drugName'], row['positive_reviews'], row['avg_rating'], row['num_reviews'], i+1), unsafe_allow_html=True)
                            review_key = f"rev_{i}"
                            if st.button(f"📝 Reviews for {row['drugName']}", key=review_key):
                                if f"reviews_visible_{review_key}" not in st.session_state:
                                    st.session_state[f"reviews_visible_{review_key}"] = True
                                    st.session_state[f"reviews_count_{review_key}"] = 3
                                else:
                                    st.session_state[f"reviews_visible_{review_key}"] = not st.session_state[f"reviews_visible_{review_key}"]
                            if st.session_state.get(f"reviews_visible_{review_key}", False):
                                reviews = condition_filtered[condition_filtered['drugName'] == row['drugName']]
                                pos_reviews = reviews[reviews['rating'] >= 7][['review', 'rating']].sort_values(by='rating', ascending=False)
                                if pos_reviews.empty:
                                    st.info("ℹ️ No positive reviews found for this drug.")
                                else:
                                    st.markdown('<div class="review-container" role="region" aria-label="Patient reviews">', unsafe_allow_html=True)
                                    for r in pos_reviews.head(st.session_state.get(f"reviews_count_{review_key}", 3)).itertuples():
                                        st.markdown(f"""
                                            <div class="review-box" role="article">
                                                <div style="color:#ffd700">{display_rating(r.rating)} {r.rating}/10</div>
                                                <p style="font-style:italic">"{r.review[:250]}{'...' if len(r.review) > 250 else ''}"</p>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    if st.session_state.get(f"reviews_count_{review_key}", 3) < len(pos_reviews):
                                        if st.button("⬇ Load More Reviews", key=f"load_more_{review_key}"):
                                            st.session_state[f"reviews_count_{review_key}"] = st.session_state.get(f"reviews_count_{review_key}", 3) + 3
                                            st.experimental_rerun()
                    with tab2:
                        for i, row in drug_stats.iloc[3:].iterrows():
                            st.markdown(create_drug_card(row['drugName'], row['positive_reviews'], row['avg_rating'], row['num_reviews']), unsafe_allow_html=True)

elif page == "🧪 Predict Review":
    st.markdown(""" 
        <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; margin-bottom:30px">
            <h2>🧪 Analyze Review Sentiment</h2>
            <p style="opacity:0.8">Use our AI to predict whether a patient review is positive or negative.</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        review_input = st.text_area("Patient Review", placeholder="e.g. This medication reduced my symptoms...", height=150, key="review")
    with col2:
        condition_input = st.selectbox("Medical Condition", 
                                      options=sorted(combined_data['condition'].dropna().unique()),
                                      index=None,
                                      placeholder="Select a condition...",
                                      key="condition_pred")
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    with st.expander("💡 Example Reviews"):
        st.markdown(""" 
            **Positive**: "This drug helped my anxiety significantly with only mild side effects."  
            **Negative**: "Experienced severe nausea and no improvement in symptoms."
        """)
    if analyze_btn and review_input and condition_input:
        with st.spinner("Analyzing sentiment..."):
            is_positive = predict_sentiment(review_input, condition_input, encoder, model)
            if is_positive is not None:
                sentiment_color = "#90ee90" if is_positive else "#ff6347"
                sentiment_icon = "✅" if is_positive else "❌"
                sentiment_text = "Positive" if is_positive else "Negative"
                sentiment_desc = "This review expresses satisfaction with the medication." if is_positive else \
                                "This review indicates dissatisfaction with the medication."
                st.markdown(f"""
                    <div class="animate-fade-in" style="background:rgba(255,255,255,0.1); backdrop-filter:blur(10px); border-radius:20px; padding:30px; text-align:center; margin:20px 0">
                        <div style="font-size:60px; margin-bottom:10px">{sentiment_icon}</div>
                        <h2 style="color:{sentiment_color}">{sentiment_text} Sentiment</h2>
                        <p>{sentiment_desc}</p>
                    </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Review Length", f"{len(review_input)} chars")
                with col2:
                    st.metric("Estimated Rating", f"{8.5 if is_positive else 3.5}/10")
                with col3:
                    st.metric("Word Count", len(review_input.split()))
            else:
                st.warning("⚠️ Unable to predict sentiment. Please check your input.")

    # ===================== USER-LEARNED AI (Sentiment) =====================
    if "user_sentiment_model" in st.session_state:
        st.markdown("---")
        st.markdown("### 🤖 Predict Sentiment (User-learned AI Model)")
        user_review = st.text_area("Paste your medicine review here (user-trained AI):", key="user_ai_review")
        if st.button("Predict Sentiment (User AI)"):
            pred = st.session_state["user_sentiment_model"].predict([user_review])[0]
            st.success(f"Prediction: {'Positive' if pred else 'Negative'} (User-learned AI)")

elif page == "📊 Analytics":
    st.markdown("""
        <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; margin-bottom:30px">
            <h2>📊 Medication Insights</h2>
            <p style="opacity:0.8">Explore trends, ratings, and sentiment across medications and conditions.</p>
        </div>
    """, unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔍 Compare", "💬 Sentiment"])
    with tab1:
        if not combined_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                top_conditions = combined_data['condition'].value_counts().head(5).reset_index()
                top_conditions.columns = ['Condition', 'Reviews']
                fig = px.bar(top_conditions, x='Condition', y='Reviews', title="Top Conditions by Reviews",
                             color='Reviews', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                top_drugs = combined_data.groupby('drugName')['rating'].agg(['mean', 'count']).sort_values('count', ascending=False).head(10).reset_index()
                top_drugs.columns = ['Drug', 'Avg Rating', 'Reviews']
                fig = px.bar(top_drugs, x='Drug', y='Avg Rating', title="Top Drugs by Rating",
                             color='Reviews', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Rating", f"{combined_data['rating'].mean():.1f}/10")
            with col2:
                st.metric("Total Reviews", f"{len(combined_data):,}")
            with col3:
                st.metric("Positive Rate", f"{(combined_data['rating'] >= 7).mean() * 100:.1f}%")
            with col4:
                st.metric("Medications", f"{combined_data['drugName'].nunique():,}")
    with tab2:
        if not combined_data.empty:
            condition = st.selectbox("Select Condition", sorted(combined_data['condition'].unique()))
            condition_data = combined_data[combined_data['condition'] == condition]
            if not condition_data.empty:
                drug_comparison = condition_data.groupby('drugName')['rating'].agg(['mean', 'count']).reset_index()
                drug_comparison = drug_comparison[drug_comparison['count'] >= 5].sort_values('mean', ascending=False)
                fig = px.bar(drug_comparison, x='drugName', y='mean', title=f"Drug Ratings for {condition}",
                             color='count', color_continuous_scale='Teal')
                st.plotly_chart(fig, use_container_width=True)
    with tab3:
        if not combined_data.empty:
            combined_data['sentiment'] = combined_data['rating'].apply(lambda x: "Positive" if x >= 7 else "Negative")
            sentiment_counts = combined_data['sentiment'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution",
                         color_discrete_sequence=['#90ee90', '#ff6347'])
            st.plotly_chart(fig, use_container_width=True)

elif page == "ℹ️ About":
    # ... (unchanged from your code) ...
    pass

elif page == "💊 Pharmacy Tools":
    # ... (unchanged from your code) ...
    pass

# ===================== FOOTER =====================
st.markdown("""
    <div style="text-align:center; padding:20px; margin-top:40px; border-top:1px solid rgba(255,255,255,0.1); font-size:14px; opacity:0.8">
        MedGuide v2.2.0 | Made with 💙 by MD Tanveer Alam | © 2025
    </div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

