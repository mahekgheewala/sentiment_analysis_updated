# Professional Sentiment Analysis Streamlit Application
# Fixed to work with the correct pickle format
# Optimized for Google Colab Environment

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pickle
import torch
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
import nltk
from textblob import TextBlob
import re
from datetime import datetime
import base64
import io
import requests
from huggingface_hub import hf_hub_download
import os

# ===========================
# STREAMLIT CONFIGURATION & CUSTOM CSS
# ===========================

def configure_app():
    """Configure Streamlit app with custom styling and settings"""
    st.set_page_config(
        page_title="AI Sentiment Analysis Studio",
        page_icon="�",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/streamlit/streamlit',
            'Report a bug': None,
            'About': "Professional Sentiment Analysis powered by Hugging Face Model"
        }
    )

def inject_custom_css():
    """Inject custom CSS for dark theme and modern UI"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Dark Theme Override */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }

    /* Main Header Styling */
    .main-header {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }

    /* Subtitle Styling */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }

    /* Custom Card Styling */
    .custom-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }

    /* Input Box Styling */
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 2px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 16px !important;
        font-family: 'Inter', sans-serif !important;
        padding: 16px !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }

    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
        outline: none !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4) !important;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.1) !important;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin: 1rem 0;
        text-align: center;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Prediction Result Cards */
    .prediction-positive {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.1) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #22c55e;
    }

    .prediction-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
    }

    .prediction-neutral {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%);
        border: 1px solid rgba(168, 85, 247, 0.3);
        color: #a855f7;
    }

    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }

    .loading-spinner {
        border: 3px solid rgba(99, 102, 241, 0.3);
        border-radius: 50%;
        border-top: 3px solid #6366f1;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .custom-card {
            padding: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# MODEL LOADING AND UTILITIES
# ===========================

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

@st.cache_resource
def load_models():
    """Load the trained pipeline model from Hugging Face with proper format handling"""
    
    # Download NLTK data first
    download_nltk_data()
    
    try:
        st.info("🔄 Loading sentiment pipeline from Hugging Face...")

        # Load from Hugging Face
        pipeline_path = hf_hub_download(
            repo_id="mahekgheewala/sentimental_analysis_updated",
            filename="sentiment_pipeline.pkl"
        )

        with open(pipeline_path, 'rb') as f:
            pipeline_data = pickle.load(f)

        # Handle both formats: tuple and dictionary
        if isinstance(pipeline_data, tuple) and len(pipeline_data) == 3:
            # New format: (xgboost_model, sentence_model, label_names)
            xgboost_model, sentence_model, label_names = pipeline_data
            st.success("✅ Pipeline loaded successfully from Hugging Face (tuple format)!")
            return xgboost_model, sentence_model, label_names, True
            
        elif isinstance(pipeline_data, dict):
            # Old format: dictionary with keys
            if 'xgboost_model' in pipeline_data and 'label_names' in pipeline_data:
                xgboost_model = pipeline_data['xgboost_model']
                label_names = pipeline_data['label_names']
                
                # Load SentenceTransformer separately (fallback method)
                try:
                    sentence_model = SentenceTransformer('paraphrase-mpnet-base-v2')
                    st.success("✅ Pipeline loaded successfully from Hugging Face (dict format)!")
                    st.info("📝 Using base SentenceTransformer model as fallback")
                    return xgboost_model, sentence_model, label_names, True
                except Exception as e:
                    st.error(f"Failed to load SentenceTransformer: {e}")
                    return None, None, None, False
            else:
                raise ValueError("Dictionary format missing required keys: 'xgboost_model' or 'label_names'")
        else:
            raise ValueError(f"Unexpected format: {type(pipeline_data)}")

    except Exception as e:
        st.error(f"❌ Error loading pipeline from Hugging Face: {e}")
        st.info("🔁 Trying to load pipeline from local file as fallback...")

        try:
            with open('sentiment_pipeline.pkl', 'rb') as f:
                pipeline_data = pickle.load(f)

            # Handle both formats for local file too
            if isinstance(pipeline_data, tuple) and len(pipeline_data) == 3:
                xgboost_model, sentence_model, label_names = pipeline_data
                st.info("📁 Loaded local pipeline successfully (tuple format).")
                return xgboost_model, sentence_model, label_names, True
                
            elif isinstance(pipeline_data, dict):
                if 'xgboost_model' in pipeline_data and 'label_names' in pipeline_data:
                    xgboost_model = pipeline_data['xgboost_model']
                    label_names = pipeline_data['label_names']
                    sentence_model = SentenceTransformer('paraphrase-mpnet-base-v2')
                    st.info("📁 Loaded local pipeline successfully (dict format).")
                    return xgboost_model, sentence_model, label_names, True
                else:
                    raise ValueError("Local dictionary format missing required keys")
            else:
                raise ValueError(f"Local file has unexpected format: {type(pipeline_data)}")

        except Exception as local_e:
            st.error(f"⚠️ Fallback also failed: {local_e}")
            return None, None, None, False


def get_sentiment_textblob(text):
    """Calculate TextBlob sentiment polarity"""
    return TextBlob(text).sentiment.polarity

def get_pos_counts(text):
    """Count POS tags (Adjectives, Nouns, Verbs)"""
    try:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)

        adj_count = sum(1 for _, tag in pos_tags if tag.startswith('J'))
        noun_count = sum(1 for _, tag in pos_tags if tag.startswith('N'))
        verb_count = sum(1 for _, tag in pos_tags if tag.startswith('V'))

        return [adj_count, noun_count, verb_count]
    except:
        return [0, 0, 0]

def predict_sentiment(text, xgboost_model, sentence_model, label_names):
    """Make sentiment prediction using the individual models"""
    if not text.strip():
        return None, None, None

    try:
        # Extract features like in the training pipeline
        # 1. Get embeddings from SentenceTransformer
        embedding = sentence_model.encode([text])
        
        # 2. Get additional features
        comment_length = np.array([len(text.split())]).reshape(-1, 1)
        sentiment_polarity = np.array([get_sentiment_textblob(text)]).reshape(-1, 1)
        pos_counts = np.array([get_pos_counts(text)])
        
        # 3. Combine all features (768 embeddings + 1 length + 1 sentiment + 3 POS = 773 features)
        features = np.hstack([embedding, comment_length, sentiment_polarity, pos_counts])
        
        # 4. Make prediction
        prediction = xgboost_model.predict(features)[0]
        probabilities = xgboost_model.predict_proba(features)[0]
        
        # 5. Get label name
        predicted_label = label_names[prediction]

        return prediction, probabilities, predicted_label

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None


# ===========================
# VISUALIZATION COMPONENTS
# ===========================

def create_confidence_chart(probabilities, label_names):
    """Create confidence score visualization"""
    if probabilities is None:
        return None

    labels = [label_names[i] for i in sorted(label_names.keys())]
    values = probabilities

    # Create color mapping
    colors = ['#ef4444', '#22c55e', '#a855f7']  # Red, Green, Purple

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f'{val:.2%}' for val in values],
            textposition='auto',
            textfont=dict(color='white', size=14, family='Inter')
        )
    ])

    fig.update_layout(
        title={
            'text': 'Confidence Scores by Sentiment',
            'x': 0.5,
            'font': {'size': 20, 'color': 'white', 'family': 'Inter'}
        },
        xaxis=dict(
            title='Sentiment',
            titlefont=dict(color='white', size=14),
            tickfont=dict(color='white', size=12)
        ),
        yaxis=dict(
            title='Confidence',
            titlefont=dict(color='white', size=14),
            tickfont=dict(color='white', size=12),
            tickformat='.0%'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter')
    )

    return fig

def create_sentiment_gauge(confidence):
    """Create sentiment confidence gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence", 'font': {'color': 'white', 'size': 20, 'family': 'Inter'}},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': 'white'},
            'bar': {'color': "#6366f1"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.3)"},
                {'range': [50, 80], 'color': "rgba(168, 85, 247, 0.3)"},
                {'range': [80, 100], 'color': "rgba(34, 197, 94, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Inter'}
    )

    return fig

# ===========================
# MAIN APPLICATION INTERFACE
# ===========================

def create_header():
    """Create the main header section"""
    st.markdown("""
    <div class="main-header">
        AI Sentiment Analysis Studio
    </div>
    <div class="subtitle">
        Professional sentiment analysis powered by Hugging Face Model: mahekgheewala/sentimental_analysis_updated
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create the sidebar with controls and information"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #6366f1; font-family: 'Inter', sans-serif; margin-bottom: 0.5rem;">
                🚀 Control Panel
            </h2>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Advanced sentiment analysis tools
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Model Information
        st.markdown("### 🤖 Model Information")
        st.info("""
        **Architecture:** Hybrid Model
        - **Source:** Hugging Face Hub
        - **Repository:** mahekgheewala/sentimental_analysis_updated
        - **Embeddings:** Sentence Transformers
        - **Classifier:** XGBoost
        - **Features:** Text + Linguistic + Semantic
        """)

        # Analysis Settings
        st.markdown("### ⚙️ Analysis Settings")

        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_features = st.checkbox("Show Feature Analysis", value=False)
        show_gauge = st.checkbox("Show Confidence Gauge", value=True)

        # Quick Actions
        st.markdown("### ⚡ Quick Actions")

        if st.button("🔄 Clear Analysis", use_container_width=True):
            # Clear relevant session state keys instead of the whole thing
            keys_to_clear = ['prediction', 'probabilities', 'predicted_label', 'text_input']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # Sample Texts
        st.markdown("### 📝 Sample Texts")
        samples = {
            "Positive": "I absolutely love this product! It exceeded all my expectations.",
            "Negative": "This is terrible. I'm extremely disappointed and frustrated.",
            "Neutral": "The weather today is partly cloudy with occasional sunshine."
        }

        for sentiment, text in samples.items():
            if st.button(f"Try {sentiment} Example", use_container_width=True):
                st.session_state.sample_text = text
                # Clear previous results when a new sample is loaded
                if 'prediction' in st.session_state:
                    del st.session_state['prediction']

        # Model Statistics
        st.markdown("### 📊 Model Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classes", "3", help="Positive, Negative, Neutral")
        with col2:
            st.metric("Features", "773", help="Embeddings + Linguistic features")

        # Hugging Face Link
        st.markdown("### 🤗 Model Repository")
        st.markdown("""
        <a href="https://huggingface.co/mahekgheewala/sentimental_analysis_updated" target="_blank"
           style="text-decoration: none;">
            <div style="background: linear-gradient(45deg, #ff7b00, #ffaa00);
                        padding: 10px; border-radius: 8px; text-align: center;
                        color: white; font-weight: bold;">
                🤗 View on Hugging Face
            </div>
        </a>
        """, unsafe_allow_html=True)

        return show_confidence, show_features, show_gauge

def create_input_section():
    """Create the text input section"""
    st.markdown("""
    <div class="custom-card">
        <h3 style="color: #6366f1; font-family: 'Inter', sans-serif; margin-bottom: 1rem; font-size: 1.5rem;">
            💬 Enter Text for Analysis
        </h3>
    """, unsafe_allow_html=True)

    # Get sample text if available
    default_text = st.session_state.get('sample_text', st.session_state.get('text_input', ''))

    # Text input
    user_text = st.text_area(
        "",
        value=default_text,
        placeholder="Enter your text here for sentiment analysis...\n\nExample: 'I had an amazing experience with this service!'",
        height=120,
        help="Enter any text and our AI will analyze its sentiment with high accuracy",
        key="text_input"
    )

    # Clear sample text after use
    if 'sample_text' in st.session_state:
        del st.session_state.sample_text

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Sentiment",
            use_container_width=True,
            help="Click to analyze the sentiment of your text"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    return user_text, analyze_button

def create_results_section(prediction, probabilities, predicted_label, show_confidence, show_features, show_gauge, user_text, label_names):
    """Create the results display section"""
    if prediction is None or probabilities is None or predicted_label is None:
        return

    # Main prediction result
    sentiment_colors = {
        'Positive': '#22c55e',
        'Negative': '#ef4444',
        'Neutral': '#a855f7'
    }

    sentiment_emojis = {
        'Positive': '😊',
        'Negative': '😞',
        'Neutral': '😐'
    }

    color = sentiment_colors.get(predicted_label, '#6366f1')
    emoji = sentiment_emojis.get(predicted_label, '🤔')
    max_confidence = max(probabilities)

    # Results header
    st.markdown("## 🎯 Analysis Results")

    # Main result card
    st.markdown(f"""
    <div class="custom-card" style="text-align: center; border: 2px solid {color};">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{emoji}</div>
        <h2 style="color: {color}; font-family: 'Inter', sans-serif; font-size: 2.5rem; margin-bottom: 0.5rem;">
            {predicted_label.upper()}
        </h2>
        <p style="color: #94a3b8; font-size: 1.2rem;">
            Confidence: <strong style="color: {color};">{max_confidence:.1%}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed analysis
    col1, col2 = st.columns(2)

    with col1:
        if show_confidence:
            st.markdown("### 📊 Confidence Distribution")
            fig_bar = create_confidence_chart(probabilities, label_names)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        if show_gauge:
            st.markdown("### 🎚️ Confidence Meter")
            fig_gauge = create_sentiment_gauge(max_confidence)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Feature analysis
    if show_features:
        st.markdown("### 🔍 Feature Analysis")

        # Text statistics
        word_count = len(user_text.split())
        char_count = len(user_text)
        sentiment_polarity = get_sentiment_textblob(user_text)
        pos_counts = get_pos_counts(user_text)

        feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)

        with feature_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{word_count}</div>
                <div class="metric-label">Words</div>
            </div>
            """, unsafe_allow_html=True)

        with feature_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{char_count}</div>
                <div class="metric-label">Characters</div>
            </div>
            """, unsafe_allow_html=True)

        with feature_col3:
            polarity_color = '#22c55e' if sentiment_polarity > 0 else '#ef4444' if sentiment_polarity < 0 else '#a855f7'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {polarity_color};">{sentiment_polarity:.2f}</div>
                <div class="metric-label">Polarity</div>
            </div>
            """, unsafe_allow_html=True)

        with feature_col4:
            total_pos = sum(pos_counts)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_pos}</div>
                <div class="metric-label">POS Tags</div>
            </div>
            """, unsafe_allow_html=True)

        # POS breakdown
        st.markdown("#### Part-of-Speech Analysis")
        pos_col1, pos_col2, pos_col3 = st.columns(3)

        with pos_col1:
            st.metric("Adjectives", pos_counts[0])
        with pos_col2:
            st.metric("Nouns", pos_counts[1])
        with pos_col3:
            st.metric("Verbs", pos_counts[2])


def main():
    """Main function to run the Streamlit app"""
    configure_app()
    inject_custom_css()
    create_header()

    # Load models and handle potential errors
    xgboost_model, sentence_model, label_names, models_loaded = load_models()

    if not models_loaded:
        st.error("🚨 Critical Error: Could not load the sentiment analysis models. The application cannot proceed.")
        st.warning("Please check your internet connection, Hugging Face repository access, or local file path.")
        return

    # Create the user interface
    show_confidence, show_features, show_gauge = create_sidebar()
    user_text, analyze_button = create_input_section()

    # Logic for triggering analysis
    if analyze_button and user_text.strip():
        with st.spinner('Analyzing sentiment...'):
            # Perform prediction
            prediction, probabilities, predicted_label = predict_sentiment(
                user_text, xgboost_model, sentence_model, label_names
            )
            # Store results in session state to persist across reruns
            if prediction is not None:
                st.session_state['prediction'] = prediction
                st.session_state['probabilities'] = probabilities
                st.session_state['predicted_label'] = predicted_label
                st.session_state['analyzed_text'] = user_text # Store the text that was analyzed
    
    # Display results if they exist in the session state
    if 'prediction' in st.session_state:
        create_results_section(
            st.session_state['prediction'],
            st.session_state['probabilities'],
            st.session_state['predicted_label'],
            show_confidence,
            show_features,
            show_gauge,
            st.session_state['analyzed_text'],
            label_names
        )

if __name__ == "__main__":
    main()
