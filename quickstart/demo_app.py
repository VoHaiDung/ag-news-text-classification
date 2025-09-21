#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive Demo Application for AG News Classification
========================================================

Streamlit-based demo application for AG News text classification,
providing an intuitive interface for model interaction.

Following UI/UX principles from:
- Nielsen (1993): "Usability Engineering"
- Shneiderman et al. (2016): "Designing the User Interface"

Author: Võ Hải Dũng
License: MIT
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configs.constants import AG_NEWS_CLASSES
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

# Page configuration
st.set_page_config(
    page_title="AG News Classifier",
    page_icon="news",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model(model_path: str):
    """
    Load model and tokenizer with caching.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def predict_text(
    text: str,
    model,
    tokenizer,
    max_length: int = 256
) -> Tuple[str, Dict[str, float]]:
    """
    Predict class for input text.
    
    Args:
        text: Input text
        model: Classification model
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (predicted_class, confidence_scores)
    """
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
    
    # Get prediction
    predicted_idx = torch.argmax(probabilities, dim=-1).item()
    predicted_class = AG_NEWS_CLASSES[predicted_idx]
    
    # Get confidence scores
    confidence_scores = {
        AG_NEWS_CLASSES[i]: float(probabilities[0, i])
        for i in range(len(AG_NEWS_CLASSES))
    }
    
    return predicted_class, confidence_scores

def create_confidence_chart(confidence_scores: Dict[str, float]):
    """Create confidence score visualization."""
    fig = go.Figure(data=[
        go.Bar(
            x=list(confidence_scores.keys()),
            y=list(confidence_scores.values()),
            marker_color=['green' if v == max(confidence_scores.values()) else 'lightblue' 
                         for v in confidence_scores.values()],
            text=[f"{v:.2%}" for v in confidence_scores.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores by Category",
        xaxis_title="Category",
        yaxis_title="Confidence",
        yaxis_range=[0, 1],
        height=400
    )
    
    return fig

def main():
    """Main application function."""
    
    # Title and description
    st.title("AG News Text Classification Demo")
    st.markdown("""
    This demo application allows you to classify news articles into four categories:
    **World**, **Sports**, **Business**, and **Sci/Tech**.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model_path = st.text_input(
            "Model Path",
            value=str(PROJECT_ROOT / "outputs" / "simple" / "model"),
            help="Path to trained model directory"
        )
        
        # Parameters
        max_length = st.slider(
            "Max Length",
            min_value=64,
            max_value=512,
            value=256,
            step=32,
            help="Maximum sequence length for tokenization"
        )
        
        # Display options
        show_confidence = st.checkbox("Show Confidence Chart", value=True)
        show_examples = st.checkbox("Show Example Texts", value=True)
    
    # Load model
    if "model" not in st.session_state:
        with st.spinner("Loading model..."):
            model, tokenizer = load_model(model_path)
            if model and tokenizer:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model. Please check the model path.")
                return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Input Text")
        
        # Text input methods
        input_method = st.radio(
            "Input Method",
            ["Type/Paste Text", "Use Example"],
            horizontal=True
        )
        
        if input_method == "Type/Paste Text":
            text_input = st.text_area(
                "Enter news article text:",
                height=200,
                placeholder="Paste or type your news article here..."
            )
        else:
            example_texts = {
                "World": "The United Nations Security Council met today to discuss the ongoing humanitarian crisis in the region. Representatives from multiple countries expressed concern over the escalating situation.",
                "Sports": "The Lakers secured a thrilling victory over the Celtics in last night's game, with LeBron James scoring 35 points and leading his team to a 110-105 win.",
                "Business": "Apple Inc. reported record quarterly earnings, beating analyst expectations with revenue of $123.9 billion, driven by strong iPhone and services sales.",
                "Sci/Tech": "Scientists at MIT have developed a new quantum computing algorithm that could revolutionize cryptography and solve complex optimization problems exponentially faster."
            }
            
            selected_example = st.selectbox(
                "Select an example:",
                list(example_texts.keys())
            )
            text_input = example_texts[selected_example]
            st.text_area("Example text:", value=text_input, height=150, disabled=True)
    
    with col2:
        st.header("Prediction")
        
        if st.button("Classify", type="primary", use_container_width=True):
            if text_input:
                with st.spinner("Classifying..."):
                    predicted_class, confidence_scores = predict_text(
                        text_input,
                        st.session_state.model,
                        st.session_state.tokenizer,
                        max_length
                    )
                
                # Display prediction
                st.success(f"**Predicted Category:** {predicted_class}")
                st.metric(
                    "Confidence",
                    f"{confidence_scores[predicted_class]:.2%}"
                )
                
                # Display all scores
                st.subheader("All Scores:")
                for category, score in sorted(
                    confidence_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    st.progress(score)
                    st.caption(f"{category}: {score:.2%}")
            else:
                st.warning("Please enter some text to classify.")
    
    # Confidence chart
    if show_confidence and text_input and st.button("Show Detailed Analysis"):
        st.header("Detailed Analysis")
        
        with st.spinner("Generating analysis..."):
            predicted_class, confidence_scores = predict_text(
                text_input,
                st.session_state.model,
                st.session_state.tokenizer,
                max_length
            )
            
            # Confidence chart
            fig = create_confidence_chart(confidence_scores)
            st.plotly_chart(fig, use_container_width=True)
            
            # Text statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Word Count", len(text_input.split()))
            
            with col2:
                st.metric("Character Count", len(text_input))
            
            with col3:
                st.metric("Sentence Count", text_input.count('.') + text_input.count('!') + text_input.count('?'))
    
    # Example texts section
    if show_examples:
        st.header("Try These Examples")
        
        examples = [
            {
                "title": "Technology News",
                "text": "Google announced a breakthrough in artificial intelligence with their new language model that can understand and generate code in multiple programming languages.",
                "expected": "Sci/Tech"
            },
            {
                "title": "Sports Update",
                "text": "The FIFA World Cup final saw an intense match between two rival teams, with the game going into extra time before a winner was decided.",
                "expected": "Sports"
            },
            {
                "title": "Business Report",
                "text": "The stock market experienced significant volatility today as investors reacted to the Federal Reserve's latest interest rate decision.",
                "expected": "Business"
            },
            {
                "title": "International News",
                "text": "World leaders gathered at the G20 summit to discuss climate change initiatives and global economic cooperation strategies.",
                "expected": "World"
            }
        ]
        
        for example in examples:
            with st.expander(f"{example['title']} (Expected: {example['expected']})"):
                st.write(example["text"])
                if st.button(f"Classify This", key=example["title"]):
                    predicted_class, confidence_scores = predict_text(
                        example["text"],
                        st.session_state.model,
                        st.session_state.tokenizer,
                        max_length
                    )
                    
                    if predicted_class == example["expected"]:
                        st.success(f"Correct! Predicted: {predicted_class} ({confidence_scores[predicted_class]:.2%})")
                    else:
                        st.error(f"Incorrect. Predicted: {predicted_class} ({confidence_scores[predicted_class]:.2%})")
    
    # Footer
    st.markdown("---")
    st.caption("AG News Text Classification Demo - Built with Streamlit")
    st.caption("For more information, see the [project documentation](https://github.com/VoHaiDung/ag-news-text-classification)")

if __name__ == "__main__":
    main()
