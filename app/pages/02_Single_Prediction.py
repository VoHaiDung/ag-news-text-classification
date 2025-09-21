"""
Single Prediction Page for AG News Classification
==================================================

Implements single text classification interface following principles from:
- Nielsen (1993): "Usability Engineering" - Response time guidelines
- Shneiderman (1984): "Response Time and Display Rate in Human Performance"

Author: Võ Hải Dũng
License: MIT
"""

import sys
from pathlib import Path
import time
from typing import Dict, Any, Optional, Tuple

# Add project root to path
PAGES_DIR = Path(__file__).parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import streamlit as st
import torch
import pandas as pd
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app import get_app_controller, get_app_config
from configs.constants import AG_NEWS_CLASSES, MAX_SEQUENCE_LENGTH
from src.utils.logging_config import setup_logging
from src.services.data_service import DataService

logger = setup_logging(__name__)

@st.cache_resource
def load_model(model_path: str) -> Tuple[Any, Any]:
    """
    Load and cache model.
    
    Args:
        model_path: Path to model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def predict_single(
    text: str,
    model: Any,
    tokenizer: Any,
    max_length: int = 256
) -> Dict[str, Any]:
    """
    Perform single text prediction.
    
    Args:
        text: Input text
        model: Classification model
        tokenizer: Tokenizer
        max_length: Max sequence length
        
    Returns:
        Prediction results dictionary
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
    
    # Process results
    predicted_idx = torch.argmax(probabilities, dim=-1).item()
    confidence_scores = {
        AG_NEWS_CLASSES[i]: float(probabilities[0, i])
        for i in range(len(AG_NEWS_CLASSES))
    }
    
    inference_time = time.time() - start_time
    
    return {
        "predicted_class": AG_NEWS_CLASSES[predicted_idx],
        "confidence": float(probabilities[0, predicted_idx]),
        "confidence_scores": confidence_scores,
        "inference_time": inference_time,
        "text_stats": {
            "word_count": len(text.split()),
            "char_count": len(text),
            "sentence_count": text.count('.') + text.count('!') + text.count('?')
        }
    }

def render_input_section() -> Optional[str]:
    """
    Render text input section.
    
    Returns:
        Input text or None
    """
    st.markdown("### Input Text")
    
    input_method = st.radio(
        "Select input method:",
        ["Direct Input", "Example Text", "File Upload"],
        horizontal=True
    )
    
    text_input = None
    
    if input_method == "Direct Input":
        text_input = st.text_area(
            "Enter news article text:",
            height=200,
            placeholder="Type or paste your news article here...",
            help="Enter the text you want to classify"
        )
    
    elif input_method == "Example Text":
        examples = {
            "World News": "The United Nations Security Council held an emergency meeting today to discuss the escalating humanitarian crisis in the region. Representatives from member nations expressed deep concern over the deteriorating situation and called for immediate action.",
            "Sports": "The championship game delivered an incredible finish as the home team scored in the final seconds to secure victory. The crowd erupted in celebration as the players embraced on the field after the dramatic win.",
            "Business": "Major technology companies reported strong quarterly earnings today, beating analyst expectations across the board. The positive results drove stock markets to new highs as investors showed renewed confidence.",
            "Technology": "Scientists have achieved a significant breakthrough in quantum computing, demonstrating a new algorithm that could solve complex problems exponentially faster than classical computers."
        }
        
        selected = st.selectbox("Choose an example:", list(examples.keys()))
        text_input = examples[selected]
        
        st.text_area(
            "Selected example:",
            value=text_input,
            height=150,
            disabled=True
        )
    
    else:  # File Upload
        uploaded_file = st.file_uploader(
            "Upload a text file:",
            type=['txt'],
            help="Upload a plain text file containing the article"
        )
        
        if uploaded_file:
            text_input = str(uploaded_file.read(), "utf-8")
            st.text_area(
                "Uploaded text (preview):",
                value=text_input[:500] + "..." if len(text_input) > 500 else text_input,
                height=150,
                disabled=True
            )
    
    return text_input

def render_results(results: Dict[str, Any]):
    """
    Render prediction results.
    
    Args:
        results: Prediction results dictionary
    """
    st.markdown("### Prediction Results")
    
    # Main result
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.success(f"**Predicted Category:** {results['predicted_class']}")
    
    with col2:
        st.metric("Confidence", f"{results['confidence']:.1%}")
    
    with col3:
        st.metric("Latency", f"{results['inference_time']*1000:.1f} ms")
    
    # Confidence visualization
    st.markdown("#### Confidence Distribution")
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(results['confidence_scores'].keys()),
            y=list(results['confidence_scores'].values()),
            marker=dict(
                color=list(results['confidence_scores'].values()),
                colorscale='Blues',
                showscale=False,
                line=dict(color='darkblue', width=1)
            ),
            text=[f"{v:.1%}" for v in results['confidence_scores'].values()],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Classification Confidence by Category",
        xaxis_title="Category",
        yaxis_title="Confidence Score",
        yaxis_range=[0, 1.1],
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed scores
    with st.expander("Detailed Scores"):
        scores_df = pd.DataFrame({
            "Category": results['confidence_scores'].keys(),
            "Confidence": [f"{v:.2%}" for v in results['confidence_scores'].values()],
            "Score": results['confidence_scores'].values()
        }).sort_values("Score", ascending=False)
        
        st.dataframe(
            scores_df[["Category", "Confidence"]],
            use_container_width=True,
            hide_index=True
        )
    
    # Text statistics
    with st.expander("Text Statistics"):
        stats = results['text_stats']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Words", stats['word_count'])
        with col2:
            st.metric("Characters", stats['char_count'])
        with col3:
            st.metric("Sentences", stats['sentence_count'])

def render_configuration() -> Dict[str, Any]:
    """
    Render configuration sidebar.
    
    Returns:
        Configuration dictionary
    """
    with st.sidebar:
        st.markdown("## Configuration")
        
        # Model selection
        st.markdown("### Model Settings")
        
        model_type = st.selectbox(
            "Model Type",
            ["DistilBERT (Fast)", "BERT (Balanced)", "RoBERTa (Accurate)"]
        )
        
        model_map = {
            "DistilBERT (Fast)": "distilbert-base-uncased",
            "BERT (Balanced)": "bert-base-uncased",
            "RoBERTa (Accurate)": "roberta-base"
        }
        
        model_path = model_map[model_type]
        
        # Processing settings
        st.markdown("### Processing")
        
        max_length = st.slider(
            "Max Length",
            min_value=64,
            max_value=512,
            value=256,
            step=32,
            help="Maximum sequence length for tokenization"
        )
        
        # Advanced settings
        with st.expander("Advanced"):
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Confidence scaling factor"
            )
            
            threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum confidence for accepting prediction"
            )
        
        return {
            "model_path": model_path,
            "max_length": max_length,
            "temperature": temperature,
            "threshold": threshold
        }

def main():
    """Main function for Single Prediction page."""
    st.set_page_config(
        page_title="Single Prediction - AG News",
        page_icon="newspaper",
        layout="wide"
    )
    
    st.markdown("# Single Text Classification")
    st.markdown("""
    Classify individual news articles into one of four categories:
    **World**, **Sports**, **Business**, or **Sci/Tech**.
    """)
    
    # Get configuration
    config = render_configuration()
    
    # Input section
    text_input = render_input_section()
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        predict_button = st.button(
            "Classify",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        clear_button = st.button(
            "Clear",
            use_container_width=True
        )
    
    if clear_button:
        st.experimental_rerun()
    
    # Perform prediction
    if predict_button and text_input:
        # Load model
        if "model" not in st.session_state or st.session_state.get("model_path") != config["model_path"]:
            with st.spinner("Loading model..."):
                model, tokenizer = load_model(config["model_path"])
                if model and tokenizer:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model_path = config["model_path"]
                else:
                    st.error("Failed to load model. Please check the model path.")
                    return
        
        # Make prediction
        with st.spinner("Classifying..."):
            results = predict_single(
                text_input,
                st.session_state.model,
                st.session_state.tokenizer,
                config["max_length"]
            )
        
        # Display results
        st.markdown("---")
        render_results(results)
        
        # Check confidence threshold
        if results["confidence"] < config["threshold"]:
            st.warning(f"Low confidence prediction ({results['confidence']:.1%} < {config['threshold']:.1%})")
    
    elif predict_button:
        st.warning("Please enter text to classify")

if __name__ == "__main__":
    main()
