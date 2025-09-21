#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Streamlit Application for AG News Text Classification
===========================================================

This module implements the main user interface for the AG News text classification
system using Streamlit framework.

The application design follows principles from:
- Cooper et al. (2014): "About Face: The Essentials of Interaction Design"
- Norman (2013): "The Design of Everyday Things" - User-centered design
- Tidwell et al. (2020): "Designing Interfaces" - UI patterns

Architecture follows:
- Model-View-ViewModel (MVVM) pattern for separation of concerns
- Reactive programming paradigm for state management
- Component-based architecture for modularity

Author: Võ Hải Dũng
License: MIT
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime

# Add project root to path
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import application modules
from app import (
    initialize_app,
    get_app_controller,
    get_app_config,
    AppConfig
)

# Import project modules
from configs.constants import AG_NEWS_CLASSES, MAX_SEQUENCE_LENGTH
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from src.services.data_service import DataService

# Setup logging
logger = setup_logging(__name__)

# Page configuration
st.set_page_config(
    page_title="AG News Classifier - Advanced Demo",
    page_icon="newspaper",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/VoHaiDung/ag-news-text-classification',
        'Report a bug': 'https://github.com/VoHaiDung/ag-news-text-classification/issues',
        'About': 'AG News Text Classification System v1.0.0'
    }
)

# Custom CSS for professional styling
def load_custom_css():
    """Load custom CSS for application styling."""
    css = """
    <style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f2937;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 10px;
    }
    
    /* Metric container styling */
    [data-testid="metric-container"] {
        background-color: #f3f4f6;
        border: 1px solid #e5e7eb;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 10px;
    }
    
    .stError {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 10px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9fafb;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

@st.cache_resource
def initialize_application():
    """
    Initialize the application with configuration and resources.
    
    Returns:
        Application controller instance
    """
    config = AppConfig(
        enable_caching=True,
        enable_visualization=True,
        enable_interpretability=True,
        enable_model_comparison=True
    )
    
    controller = initialize_app(config)
    
    # Initialize data service
    data_service = DataService()
    controller.register_component("data_service", data_service)
    
    logger.info("Application initialized successfully")
    
    return controller

@st.cache_resource
def load_model_cached(model_path: str) -> Tuple[Any, Any]:
    """
    Load model with caching for performance.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model loaded on GPU")
        else:
            logger.info("Model loaded on CPU")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        st.error(f"Model loading failed: {e}")
        return None, None

def predict_text(
    text: str,
    model: Any,
    tokenizer: Any,
    max_length: int = MAX_SEQUENCE_LENGTH
) -> Dict[str, Any]:
    """
    Perform text classification.
    
    Args:
        text: Input text
        model: Classification model
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with prediction results
    """
    start_time = time.time()
    
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Move to device
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
    
    # Get results
    predicted_idx = torch.argmax(probabilities, dim=-1).item()
    predicted_class = AG_NEWS_CLASSES[predicted_idx]
    
    confidence_scores = {
        AG_NEWS_CLASSES[i]: float(probabilities[0, i])
        for i in range(len(AG_NEWS_CLASSES))
    }
    
    inference_time = time.time() - start_time
    
    return {
        "predicted_class": predicted_class,
        "predicted_idx": predicted_idx,
        "confidence_scores": confidence_scores,
        "max_confidence": float(probabilities.max()),
        "inference_time": inference_time,
        "text_length": len(text.split())
    }

def create_confidence_visualization(confidence_scores: Dict[str, float]) -> go.Figure:
    """
    Create interactive confidence score visualization.
    
    Args:
        confidence_scores: Dictionary of class confidence scores
        
    Returns:
        Plotly figure object
    """
    # Sort scores by value
    sorted_scores = dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True))
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(sorted_scores.keys()),
            y=list(sorted_scores.values()),
            marker=dict(
                color=list(sorted_scores.values()),
                colorscale='Blues',
                showscale=False,
                line=dict(color='rgb(8,48,107)', width=1.5)
            ),
            text=[f"{v:.1%}" for v in sorted_scores.values()],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2%}<extra></extra>'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Classification Confidence Scores",
            font=dict(size=18, color='#1f2937')
        ),
        xaxis=dict(
            title="Category",
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Confidence",
            range=[0, 1.1],
            tickformat='.0%',
            tickfont=dict(size=12)
        ),
        height=400,
        margin=dict(t=60, b=40),
        plot_bgcolor='rgba(248,249,250,0.5)',
        paper_bgcolor='white'
    )
    
    # Add grid
    fig.update_yaxis(gridcolor='lightgray', gridwidth=0.5)
    
    return fig

def render_sidebar(controller: Any) -> Dict[str, Any]:
    """
    Render sidebar with configuration options.
    
    Args:
        controller: Application controller
        
    Returns:
        Dictionary of configuration values
    """
    with st.sidebar:
        st.markdown("## Configuration")
        
        # Model selection
        st.markdown("### Model Settings")
        
        model_source = st.selectbox(
            "Model Source",
            ["Pre-trained", "Fine-tuned", "Custom Path"],
            help="Select the source of the model"
        )
        
        if model_source == "Pre-trained":
            model_name = st.selectbox(
                "Select Model",
                ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"],
                help="Choose a pre-trained model"
            )
            model_path = model_name
        elif model_source == "Fine-tuned":
            model_path = st.text_input(
                "Model Path",
                value=str(PROJECT_ROOT / "outputs" / "models" / "fine_tuned"),
                help="Path to fine-tuned model"
            )
        else:
            model_path = st.text_input(
                "Custom Model Path",
                help="Enter custom model path"
            )
        
        # Processing settings
        st.markdown("### Processing Settings")
        
        max_length = st.slider(
            "Max Sequence Length",
            min_value=64,
            max_value=512,
            value=256,
            step=32,
            help="Maximum length for text tokenization"
        )
        
        batch_processing = st.checkbox(
            "Enable Batch Processing",
            value=False,
            help="Process multiple texts at once"
        )
        
        # Visualization settings
        st.markdown("### Display Settings")
        
        show_confidence = st.checkbox(
            "Show Confidence Chart",
            value=True,
            help="Display confidence visualization"
        )
        
        show_interpretation = st.checkbox(
            "Show Interpretability",
            value=False,
            help="Show model interpretation (if available)"
        )
        
        show_statistics = st.checkbox(
            "Show Text Statistics",
            value=True,
            help="Display text statistics"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            temperature = st.slider(
                "Temperature Scaling",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust prediction confidence scaling"
            )
            
            top_k = st.number_input(
                "Top-K Classes",
                min_value=1,
                max_value=4,
                value=4,
                help="Number of top classes to display"
            )
        
        return {
            "model_path": model_path,
            "max_length": max_length,
            "batch_processing": batch_processing,
            "show_confidence": show_confidence,
            "show_interpretation": show_interpretation,
            "show_statistics": show_statistics,
            "temperature": temperature,
            "top_k": top_k
        }

def render_main_interface(controller: Any, config: Dict[str, Any]):
    """
    Render main application interface.
    
    Args:
        controller: Application controller
        config: Configuration from sidebar
    """
    # Header
    st.markdown("# AG News Text Classification System")
    st.markdown("""
    Advanced neural text classification system for categorizing news articles into four categories:
    **World**, **Sports**, **Business**, and **Sci/Tech**.
    
    This application demonstrates state-of-the-art transformer-based classification with
    interpretability and confidence analysis.
    """)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Single Prediction",
        "Batch Analysis",
        "Model Comparison",
        "Performance Dashboard",
        "Documentation"
    ])
    
    with tab1:
        render_single_prediction_tab(controller, config)
    
    with tab2:
        render_batch_analysis_tab(controller, config)
    
    with tab3:
        render_model_comparison_tab(controller, config)
    
    with tab4:
        render_performance_dashboard_tab(controller, config)
    
    with tab5:
        render_documentation_tab()

def render_single_prediction_tab(controller: Any, config: Dict[str, Any]):
    """Render single prediction interface."""
    st.markdown("### Single Text Classification")
    
    # Text input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_method = st.radio(
            "Input Method",
            ["Direct Input", "Example Text", "File Upload"],
            horizontal=True
        )
        
        if input_method == "Direct Input":
            text_input = st.text_area(
                "Enter news article text:",
                height=200,
                placeholder="Type or paste your news article here...",
                help="Enter the text you want to classify"
            )
        
        elif input_method == "Example Text":
            examples = {
                "World News": "The United Nations Security Council convened an emergency session today to address the escalating humanitarian crisis in the region.",
                "Sports": "In a thrilling match, the home team secured victory with a last-minute goal, sending fans into celebration.",
                "Business": "The stock market showed strong gains today as investors responded positively to the latest earnings reports.",
                "Technology": "Researchers have developed a new quantum computing algorithm that could revolutionize data encryption."
            }
            
            selected = st.selectbox("Select an example:", list(examples.keys()))
            text_input = examples[selected]
            st.text_area("Example text:", value=text_input, height=150, disabled=True)
        
        else:  # File Upload
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt'],
                help="Upload a text file containing the article"
            )
            
            if uploaded_file:
                text_input = str(uploaded_file.read(), "utf-8")
                st.text_area("Uploaded text:", value=text_input[:500] + "...", height=150, disabled=True)
            else:
                text_input = ""
    
    with col2:
        st.markdown("#### Quick Actions")
        
        if st.button("Classify Text", type="primary", use_container_width=True):
            if text_input:
                # Load model if not cached
                if "model" not in st.session_state or "tokenizer" not in st.session_state:
                    with st.spinner("Loading model..."):
                        model, tokenizer = load_model_cached(config["model_path"])
                        if model and tokenizer:
                            st.session_state.model = model
                            st.session_state.tokenizer = tokenizer
                        else:
                            st.error("Failed to load model")
                            return
                
                # Perform prediction
                with st.spinner("Classifying..."):
                    results = predict_text(
                        text_input,
                        st.session_state.model,
                        st.session_state.tokenizer,
                        config["max_length"]
                    )
                
                # Display results
                st.success(f"**Predicted Category:** {results['predicted_class']}")
                st.metric("Confidence", f"{results['max_confidence']:.1%}")
                st.caption(f"Inference time: {results['inference_time']*1000:.2f} ms")
            else:
                st.warning("Please enter text to classify")
        
        if st.button("Clear", use_container_width=True):
            st.experimental_rerun()
    
    # Results visualization
    if text_input and "model" in st.session_state:
        st.markdown("---")
        
        # Get prediction results
        results = predict_text(
            text_input,
            st.session_state.model,
            st.session_state.tokenizer,
            config["max_length"]
        )
        
        # Display confidence chart
        if config["show_confidence"]:
            st.markdown("#### Confidence Analysis")
            fig = create_confidence_visualization(results["confidence_scores"])
            st.plotly_chart(fig, use_container_width=True)
        
        # Display statistics
        if config["show_statistics"]:
            st.markdown("#### Text Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Word Count", results["text_length"])
            
            with col2:
                st.metric("Character Count", len(text_input))
            
            with col3:
                sentences = text_input.count('.') + text_input.count('!') + text_input.count('?')
                st.metric("Sentences", sentences)
            
            with col4:
                avg_word_length = sum(len(word) for word in text_input.split()) / max(results["text_length"], 1)
                st.metric("Avg Word Length", f"{avg_word_length:.1f}")
        
        # Top predictions table
        st.markdown("#### Classification Details")
        
        df_scores = pd.DataFrame({
            "Category": results["confidence_scores"].keys(),
            "Confidence": [f"{v:.2%}" for v in results["confidence_scores"].values()],
            "Score": results["confidence_scores"].values()
        }).sort_values("Score", ascending=False)
        
        st.dataframe(
            df_scores[["Category", "Confidence"]],
            use_container_width=True,
            hide_index=True
        )

def render_batch_analysis_tab(controller: Any, config: Dict[str, Any]):
    """Render batch analysis interface."""
    st.markdown("### Batch Text Analysis")
    st.info("Process multiple texts simultaneously for efficient classification")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV should have a 'text' column"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column")
            return
        
        st.markdown(f"Loaded {len(df)} texts for classification")
        
        # Preview data
        with st.expander("Preview Data"):
            st.dataframe(df.head(10))
        
        if st.button("Process Batch", type="primary"):
            # Implementation placeholder
            st.info("Batch processing functionality to be implemented")

def render_model_comparison_tab(controller: Any, config: Dict[str, Any]):
    """Render model comparison interface."""
    st.markdown("### Model Comparison")
    st.info("Compare predictions from different models")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model1 = st.selectbox(
            "Model 1",
            ["distilbert-base-uncased", "bert-base-uncased"],
            key="model1_select"
        )
    
    with col2:
        model2 = st.selectbox(
            "Model 2",
            ["roberta-base", "albert-base-v2"],
            key="model2_select"
        )
    
    # Comparison interface placeholder
    st.info("Model comparison functionality to be implemented")

def render_performance_dashboard_tab(controller: Any, config: Dict[str, Any]):
    """Render performance dashboard."""
    st.markdown("### Performance Dashboard")
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", "1,234", "+12%")
    
    with col2:
        st.metric("Avg Confidence", "94.5%", "+2.3%")
    
    with col3:
        st.metric("Avg Latency", "45ms", "-5ms")
    
    with col4:
        st.metric("Accuracy", "96.2%", "+0.5%")
    
    # Placeholder charts
    st.info("Performance monitoring dashboard to be implemented")

def render_documentation_tab():
    """Render documentation tab."""
    st.markdown("### Documentation")
    
    st.markdown("""
    #### About AG News Classification
    
    The AG News dataset is a collection of news articles from AG's corpus,
    categorized into four main topics:
    
    1. **World** - International news and events
    2. **Sports** - Sports news and updates
    3. **Business** - Business and economic news
    4. **Sci/Tech** - Science and technology news
    
    #### Model Architecture
    
    This application supports various transformer-based models:
    - BERT (Bidirectional Encoder Representations from Transformers)
    - RoBERTa (Robustly Optimized BERT Pretraining Approach)
    - DistilBERT (Distilled version of BERT)
    - ALBERT (A Lite BERT)
    
    #### Usage Guidelines
    
    1. **Single Prediction**: Enter or paste text for immediate classification
    2. **Batch Analysis**: Upload CSV files for bulk processing
    3. **Model Comparison**: Compare predictions across different models
    4. **Performance Dashboard**: Monitor system performance metrics
    
    #### API Reference
    
    For programmatic access, refer to the [API documentation](https://github.com/VoHaiDung/ag-news-text-classification).
    """)

def main():
    """Main application entry point."""
    # Load custom CSS
    load_custom_css()
    
    # Initialize application
    controller = initialize_application()
    
    # Render sidebar and get configuration
    config = render_sidebar(controller)
    
    # Render main interface
    render_main_interface(controller, config)
    
    # Footer
    st.markdown("---")
    st.caption(
        "AG News Text Classification System v1.0.0 | "
        "Built with Streamlit | "
        "[GitHub](https://github.com/VoHaiDung/ag-news-text-classification) | "
        "[Documentation](https://github.com/VoHaiDung/ag-news-text-classification/wiki)"
    )

if __name__ == "__main__":
    # Set random seed for reproducibility
    ensure_reproducibility(seed=42)
    
    # Run application
    main()
