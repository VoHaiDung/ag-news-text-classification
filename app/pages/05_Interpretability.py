"""
Interpretability Page for AG News Classification
=================================================

Implements model interpretability interface following principles from:
- Ribeiro et al. (2016): "Why Should I Trust You? Explaining the Predictions of Any Classifier"
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
- Sundararajan et al. (2017): "Axiomatic Attribution for Deep Networks"

Author: Võ Hải Dũng
License: MIT
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time

# Add project root to path
PAGES_DIR = Path(__file__).parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app import get_app_controller
from configs.constants import AG_NEWS_CLASSES, MAX_SEQUENCE_LENGTH
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@st.cache_resource
def load_model_for_interpretation(model_path: str) -> Tuple[Any, Any]:
    """
    Load model with gradient support for interpretation.
    
    Args:
        model_path: Path to model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_attentions=True,
            output_hidden_states=True
        )
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def compute_attention_scores(
    text: str,
    model: Any,
    tokenizer: Any,
    layer: int = -1
) -> Dict[str, Any]:
    """
    Compute attention scores for interpretation.
    
    Args:
        text: Input text
        model: Model with attention outputs
        tokenizer: Tokenizer
        layer: Which layer's attention to use
        
    Returns:
        Dictionary with attention data
    """
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    
    # Get tokens for display
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Move to device
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Forward pass with attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
    
    # Get attention from specified layer
    attention_layer = attentions[layer][0]  # Shape: [num_heads, seq_len, seq_len]
    
    # Average over heads
    avg_attention = attention_layer.mean(dim=0).cpu().numpy()
    
    # Get CLS token attention (for classification)
    cls_attention = avg_attention[0, :]
    
    # Filter out padding
    valid_length = (inputs["attention_mask"][0] == 1).sum().item()
    cls_attention = cls_attention[:valid_length]
    valid_tokens = tokens[:valid_length]
    
    return {
        "tokens": valid_tokens,
        "attention_scores": cls_attention,
        "full_attention": avg_attention[:valid_length, :valid_length],
        "predictions": probs[0].cpu().numpy()
    }

def compute_integrated_gradients(
    text: str,
    model: Any,
    tokenizer: Any,
    target_class: int,
    n_steps: int = 50
) -> Dict[str, np.ndarray]:
    """
    Compute Integrated Gradients for feature attribution.
    
    Based on Sundararajan et al. (2017): "Axiomatic Attribution for Deep Networks"
    
    Args:
        text: Input text
        model: Model
        tokenizer: Tokenizer
        target_class: Target class for attribution
        n_steps: Number of integration steps
        
    Returns:
        Attribution scores
    """
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Get embeddings
    embeddings = model.get_input_embeddings()
    input_embeds = embeddings(inputs["input_ids"])
    
    # Create baseline (zero embeddings)
    baseline = torch.zeros_like(input_embeds)
    
    # Compute integrated gradients
    alphas = torch.linspace(0, 1, n_steps).view(-1, 1, 1)
    if torch.cuda.is_available():
        alphas = alphas.cuda()
    
    # Interpolate between baseline and input
    interpolated = baseline + alphas * (input_embeds - baseline)
    
    gradients = []
    for i in range(n_steps):
        interp_input = interpolated[i:i+1]
        interp_input.requires_grad = True
        
        # Forward pass
        outputs = model(inputs_embeds=interp_input, attention_mask=inputs["attention_mask"])
        score = outputs.logits[0, target_class]
        
        # Backward pass
        model.zero_grad()
        score.backward()
        
        gradients.append(interp_input.grad.cpu().numpy())
    
    # Average gradients
    avg_gradients = np.mean(gradients, axis=0)
    
    # Compute integrated gradients
    integrated_gradients = (input_embeds.cpu().numpy() - baseline.cpu().numpy()) * avg_gradients
    
    # Sum over embedding dimension
    attribution_scores = np.sum(np.abs(integrated_gradients), axis=-1)[0]
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    valid_length = (inputs["attention_mask"][0] == 1).sum().item()
    
    return {
        "tokens": tokens[:valid_length],
        "attributions": attribution_scores[:valid_length]
    }

def render_attention_visualization(attention_data: Dict[str, Any]):
    """
    Render attention visualization.
    
    Args:
        attention_data: Dictionary with attention scores
    """
    st.markdown("#### Attention Weights")
    
    # Token-level attention
    tokens = attention_data["tokens"]
    scores = attention_data["attention_scores"]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=tokens,
            y=scores,
            marker=dict(
                color=scores,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Attention")
            ),
            hovertemplate='Token: %{x}<br>Attention: %{y:.3f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Token Attention Scores",
        xaxis_title="Tokens",
        yaxis_title="Attention Weight",
        height=400,
        xaxis=dict(tickangle=-45)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Attention heatmap
    with st.expander("Attention Matrix"):
        fig = px.imshow(
            attention_data["full_attention"],
            labels=dict(x="To Token", y="From Token", color="Attention"),
            x=tokens,
            y=tokens,
            color_continuous_scale="Blues",
            title="Full Attention Matrix"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def render_feature_importance(
    text: str,
    tokens: List[str],
    importance_scores: np.ndarray,
    prediction: str
):
    """
    Render feature importance visualization.
    
    Args:
        text: Original text
        tokens: List of tokens
        importance_scores: Importance scores for each token
        prediction: Model prediction
    """
    st.markdown("#### Feature Importance")
    
    # Normalize scores
    normalized_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min() + 1e-8)
    
    # Create highlighted text
    html_parts = []
    for token, score in zip(tokens, normalized_scores):
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        
        # Color intensity based on importance
        opacity = min(0.3 + score * 0.7, 1.0)
        color = f"rgba(255, 0, 0, {opacity})"
        
        # Clean token
        clean_token = token.replace("##", "")
        
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px; margin: 1px; '
            f'border-radius: 3px;" title="Importance: {score:.3f}">{clean_token}</span>'
        )
    
    # Display highlighted text
    st.markdown("**Text with importance highlighting:**")
    st.markdown(" ".join(html_parts), unsafe_allow_html=True)
    
    # Top important tokens
    top_indices = np.argsort(importance_scores)[-10:][::-1]
    top_tokens = [tokens[i] for i in top_indices if tokens[i] not in ["[CLS]", "[SEP]", "[PAD]"]]
    top_scores = [importance_scores[i] for i in top_indices if tokens[i] not in ["[CLS]", "[SEP]", "[PAD]"]]
    
    # Bar chart of top tokens
    fig = go.Figure(data=[
        go.Bar(
            x=top_tokens[:10],
            y=top_scores[:10],
            marker_color='coral'
        )
    ])
    
    fig.update_layout(
        title=f"Top Important Tokens for '{prediction}' Prediction",
        xaxis_title="Token",
        yaxis_title="Importance Score",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_configuration() -> Dict[str, Any]:
    """Render configuration sidebar."""
    with st.sidebar:
        st.markdown("## Configuration")
        
        # Model selection
        st.markdown("### Model")
        
        model_type = st.selectbox(
            "Select Model",
            ["BERT", "RoBERTa", "DistilBERT"]
        )
        
        model_map = {
            "BERT": "bert-base-uncased",
            "RoBERTa": "roberta-base",
            "DistilBERT": "distilbert-base-uncased"
        }
        
        # Interpretation method
        st.markdown("### Method")
        
        method = st.selectbox(
            "Interpretation Method",
            ["Attention Weights", "Integrated Gradients", "Both"]
        )
        
        # Visualization settings
        st.markdown("### Visualization")
        
        show_raw_scores = st.checkbox("Show raw scores", value=False)
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Reds", "Blues", "Viridis", "Plasma"]
        )
        
        # Advanced
        with st.expander("Advanced"):
            attention_layer = st.slider(
                "Attention Layer",
                min_value=-12,
                max_value=-1,
                value=-1,
                help="Which layer's attention to visualize"
            )
            
            n_integration_steps = st.slider(
                "Integration Steps",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                help="Number of steps for integrated gradients"
            )
        
        return {
            "model_path": model_map[model_type],
            "method": method,
            "show_raw_scores": show_raw_scores,
            "color_scheme": color_scheme,
            "attention_layer": attention_layer,
            "n_integration_steps": n_integration_steps
        }

def main():
    """Main function for Interpretability page."""
    st.set_page_config(
        page_title="Interpretability - AG News",
        page_icon="🔍",
        layout="wide"
    )
    
    st.markdown("# Model Interpretability")
    st.markdown("""
    Understand how the model makes predictions through attention visualization
    and feature attribution methods.
    """)
    
    # Get configuration
    config = render_configuration()
    
    # Text input
    st.markdown("### Input Text")
    
    text_input = st.text_area(
        "Enter news article text:",
        height=150,
        placeholder="Type or paste your news article here..."
    )
    
    # Analyze button
    if st.button("Analyze", type="primary"):
        if text_input:
            # Load model
            with st.spinner("Loading model..."):
                model, tokenizer = load_model_for_interpretation(config["model_path"])
                
                if not model or not tokenizer:
                    st.error("Failed to load model")
                    return
            
            # Get predictions first
            with st.spinner("Making prediction..."):
                inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=256)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    predicted_class = AG_NEWS_CLASSES[torch.argmax(probs).item()]
                    confidence = torch.max(probs).item()
            
            # Display prediction
            st.success(f"**Prediction:** {predicted_class} (Confidence: {confidence:.1%})")
            
            st.markdown("---")
            
            # Perform interpretation
            if config["method"] in ["Attention Weights", "Both"]:
                with st.spinner("Computing attention weights..."):
                    attention_data = compute_attention_scores(
                        text_input,
                        model,
                        tokenizer,
                        config["attention_layer"]
                    )
                
                render_attention_visualization(attention_data)
            
            if config["method"] in ["Integrated Gradients", "Both"]:
                with st.spinner("Computing integrated gradients..."):
                    target_class = AG_NEWS_CLASSES.index(predicted_class)
                    ig_data = compute_integrated_gradients(
                        text_input,
                        model,
                        tokenizer,
                        target_class,
                        config["n_integration_steps"]
                    )
                
                render_feature_importance(
                    text_input,
                    ig_data["tokens"],
                    ig_data["attributions"],
                    predicted_class
                )
            
            # Raw scores
            if config["show_raw_scores"]:
                with st.expander("Raw Scores"):
                    if config["method"] in ["Attention Weights", "Both"]:
                        st.json({
                            "attention_scores": attention_data["attention_scores"].tolist(),
                            "tokens": attention_data["tokens"]
                        })
                    
                    if config["method"] in ["Integrated Gradients", "Both"]:
                        st.json({
                            "attribution_scores": ig_data["attributions"].tolist(),
                            "tokens": ig_data["tokens"]
                        })
        else:
            st.warning("Please enter text to analyze")

if __name__ == "__main__":
    main()
