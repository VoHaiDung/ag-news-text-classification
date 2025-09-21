"""
Model Comparison Page for AG News Classification
=================================================

Implements model comparison interface following principles from:
- Plaisant (2004): "The Challenge of Information Visualization Evaluation"
- North (2006): "Toward Measuring Visualization Insight"

Author: Võ Hải Dũng
License: MIT
"""

import sys
from pathlib import Path
import time
from typing import Dict, List, Any, Tuple

# Add project root to path
PAGES_DIR = Path(__file__).parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app import get_app_controller
from configs.constants import AG_NEWS_CLASSES
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@st.cache_resource
def load_models(model_names: List[str]) -> Dict[str, Tuple[Any, Any]]:
    """
    Load multiple models for comparison.
    
    Args:
        model_names: List of model names
        
    Returns:
        Dictionary of model name to (model, tokenizer) tuple
    """
    models = {}
    
    for name in model_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name)
            model.eval()
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            models[name] = (model, tokenizer)
            logger.info(f"Loaded model: {name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            st.error(f"Failed to load {name}")
    
    return models

def predict_with_models(
    text: str,
    models: Dict[str, Tuple[Any, Any]],
    max_length: int = 256
) -> pd.DataFrame:
    """
    Get predictions from multiple models.
    
    Args:
        text: Input text
        models: Dictionary of models
        max_length: Max sequence length
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, (model, tokenizer) in models.items():
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
            probs = torch.softmax(logits, dim=-1)
        
        # Get results
        predicted_idx = torch.argmax(probs, dim=-1).item()
        confidence = float(probs[0, predicted_idx])
        inference_time = time.time() - start_time
        
        # Store results
        result = {
            "Model": model_name.split("/")[-1],
            "Prediction": AG_NEWS_CLASSES[predicted_idx],
            "Confidence": confidence,
            "Latency (ms)": inference_time * 1000
        }
        
        # Add probability for each class
        for i, class_name in enumerate(AG_NEWS_CLASSES):
            result[f"P({class_name})"] = float(probs[0, i])
        
        results.append(result)
    
    return pd.DataFrame(results)

def render_model_selection() -> List[str]:
    """
    Render model selection interface.
    
    Returns:
        List of selected model names
    """
    st.markdown("### Model Selection")
    
    available_models = {
        "DistilBERT": "distilbert-base-uncased",
        "BERT Base": "bert-base-uncased",
        "RoBERTa Base": "roberta-base",
        "ALBERT Base": "albert-base-v2"
    }
    
    col1, col2 = st.columns(2)
    
    selected_models = []
    
    with col1:
        st.markdown("#### Available Models")
        for display_name, model_name in list(available_models.items())[:2]:
            if st.checkbox(display_name, value=True, key=f"model_{display_name}"):
                selected_models.append(model_name)
    
    with col2:
        st.markdown("#### Additional Models")
        for display_name, model_name in list(available_models.items())[2:]:
            if st.checkbox(display_name, value=False, key=f"model_{display_name}"):
                selected_models.append(model_name)
    
    if len(selected_models) < 2:
        st.warning("Please select at least 2 models for comparison")
    
    return selected_models

def render_comparison_results(results_df: pd.DataFrame, text: str):
    """
    Render model comparison results.
    
    Args:
        results_df: DataFrame with comparison results
        text: Original input text
    """
    st.markdown("### Comparison Results")
    
    # Summary table
    st.markdown("#### Summary")
    
    summary_cols = ["Model", "Prediction", "Confidence", "Latency (ms)"]
    summary_df = results_df[summary_cols].copy()
    summary_df["Confidence"] = summary_df["Confidence"].apply(lambda x: f"{x:.1%}")
    summary_df["Latency (ms)"] = summary_df["Latency (ms)"].apply(lambda x: f"{x:.1f}")
    
    # Highlight best values
    st.dataframe(
        summary_df.style.highlight_max(
            subset=["Confidence"],
            props='background-color: lightgreen'
        ).highlight_min(
            subset=["Latency (ms)"],
            props='background-color: lightblue'
        ),
        use_container_width=True,
        hide_index=True
    )
    
    # Agreement analysis
    predictions = results_df["Prediction"].values
    unique_predictions = np.unique(predictions)
    
    if len(unique_predictions) == 1:
        st.success(f"All models agree: **{unique_predictions[0]}**")
    else:
        st.warning(f"Models disagree: {', '.join(unique_predictions)}")
        
        # Show vote counts
        vote_counts = pd.Series(predictions).value_counts()
        st.markdown("**Voting Results:**")
        for pred, count in vote_counts.items():
            st.write(f"- {pred}: {count} vote(s)")
    
    # Visualizations
    st.markdown("#### Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Confidence Comparison", "Probability Distribution", "Performance"])
    
    with tab1:
        # Confidence comparison bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=results_df["Model"],
                y=results_df["Confidence"],
                marker_color=['green' if p == results_df["Prediction"].mode()[0] else 'lightblue' 
                             for p in results_df["Prediction"]],
                text=[f"{c:.1%}" for c in results_df["Confidence"]],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Model Confidence Comparison",
            xaxis_title="Model",
            yaxis_title="Confidence",
            yaxis_range=[0, 1.1],
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Probability distribution heatmap
        prob_cols = [f"P({c})" for c in AG_NEWS_CLASSES]
        prob_data = results_df[prob_cols].values
        
        fig = go.Figure(data=go.Heatmap(
            z=prob_data,
            x=AG_NEWS_CLASSES,
            y=results_df["Model"],
            colorscale='Blues',
            text=[[f"{v:.1%}" for v in row] for row in prob_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Probability")
        ))
        
        fig.update_layout(
            title="Probability Distribution Across Models",
            xaxis_title="Category",
            yaxis_title="Model",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Latency comparison
            fig = px.bar(
                results_df,
                x="Model",
                y="Latency (ms)",
                title="Inference Latency",
                color="Latency (ms)",
                color_continuous_scale="Reds"
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence vs Latency scatter
            fig = px.scatter(
                results_df,
                x="Latency (ms)",
                y="Confidence",
                text="Model",
                title="Confidence vs Latency Trade-off",
                size_max=15
            )
            
            fig.update_traces(textposition='top center')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

def render_configuration() -> Dict[str, Any]:
    """Render configuration sidebar."""
    with st.sidebar:
        st.markdown("## Configuration")
        
        # Processing settings
        st.markdown("### Processing")
        
        max_length = st.slider(
            "Max Length",
            min_value=64,
            max_value=512,
            value=256,
            step=32
        )
        
        # Comparison settings
        st.markdown("### Comparison")
        
        show_probabilities = st.checkbox(
            "Show all probabilities",
            value=True
        )
        
        highlight_disagreement = st.checkbox(
            "Highlight disagreements",
            value=True
        )
        
        # Advanced
        with st.expander("Advanced"):
            ensemble_method = st.selectbox(
                "Ensemble Method",
                ["Majority Voting", "Weighted Average", "Maximum Confidence"]
            )
            
            confidence_weighting = st.checkbox(
                "Use confidence weighting",
                value=False
            )
        
        return {
            "max_length": max_length,
            "show_probabilities": show_probabilities,
            "highlight_disagreement": highlight_disagreement,
            "ensemble_method": ensemble_method,
            "confidence_weighting": confidence_weighting
        }

def main():
    """Main function for Model Comparison page."""
    st.set_page_config(
        page_title="Model Comparison - AG News",
        page_icon="newspaper",
        layout="wide"
    )
    
    st.markdown("# Model Comparison")
    st.markdown("""
    Compare predictions from multiple models to understand their agreement
    and performance characteristics.
    """)
    
    # Get configuration
    config = render_configuration()
    
    # Model selection
    selected_models = render_model_selection()
    
    if len(selected_models) >= 2:
        # Text input
        st.markdown("### Input Text")
        
        text_input = st.text_area(
            "Enter news article text:",
            height=150,
            placeholder="Type or paste your news article here..."
        )
        
        # Compare button
        if st.button("Compare Models", type="primary"):
            if text_input:
                # Load models
                with st.spinner(f"Loading {len(selected_models)} models..."):
                    models = load_models(selected_models)
                
                if len(models) >= 2:
                    # Get predictions
                    with st.spinner("Getting predictions..."):
                        results_df = predict_with_models(
                            text_input,
                            models,
                            config["max_length"]
                        )
                    
                    # Display results
                    st.markdown("---")
                    render_comparison_results(results_df, text_input)
                    
                    # Ensemble prediction
                    if config["ensemble_method"] == "Majority Voting":
                        ensemble_pred = results_df["Prediction"].mode()[0]
                        st.info(f"Ensemble Prediction (Majority Vote): **{ensemble_pred}**")
                    
                    elif config["ensemble_method"] == "Maximum Confidence":
                        max_conf_idx = results_df["Confidence"].idxmax()
                        ensemble_pred = results_df.loc[max_conf_idx, "Prediction"]
                        st.info(f"Ensemble Prediction (Max Confidence): **{ensemble_pred}**")
                else:
                    st.error("Failed to load required models")
            else:
                st.warning("Please enter text to classify")

if __name__ == "__main__":
    main()
