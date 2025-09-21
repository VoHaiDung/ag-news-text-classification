"""
Batch Analysis Page for AG News Classification
===============================================

Implements batch processing interface following principles from:
- Card et al. (1983): "The Psychology of Human-Computer Interaction"
- Dix et al. (2004): "Human-Computer Interaction" - Batch processing patterns

Author: Võ Hải Dũng
License: MIT
"""

import sys
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
import io

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
from torch.utils.data import DataLoader, Dataset
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app import get_app_controller
from configs.constants import AG_NEWS_CLASSES
from src.utils.logging_config import setup_logging
from src.services.data_service import DataService

logger = setup_logging(__name__)

class BatchDataset(Dataset):
    """Dataset for batch processing."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "idx": idx
        }

@st.cache_resource
def load_model(model_path: str):
    """Load and cache model."""
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

def process_batch(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    batch_size: int = 32,
    max_length: int = 256,
    progress_callback=None
) -> pd.DataFrame:
    """
    Process batch of texts.
    
    Args:
        texts: List of texts to classify
        model: Classification model
        tokenizer: Tokenizer
        batch_size: Batch size for processing
        max_length: Max sequence length
        progress_callback: Progress update callback
        
    Returns:
        DataFrame with results
    """
    dataset = BatchDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_confidences = []
    all_probs = []
    
    device = next(model.parameters()).device
    
    total_batches = len(dataloader)
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Update progress
        if progress_callback:
            progress_callback((batch_idx + 1) / total_batches)
        
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get predictions
        predictions = torch.argmax(probs, dim=-1)
        confidences = torch.max(probs, dim=-1).values
        
        all_predictions.extend(predictions.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    processing_time = time.time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "text": texts,
        "predicted_label": all_predictions,
        "predicted_class": [AG_NEWS_CLASSES[p] for p in all_predictions],
        "confidence": all_confidences
    })
    
    # Add probability columns
    for i, class_name in enumerate(AG_NEWS_CLASSES):
        results_df[f"prob_{class_name}"] = [p[i] for p in all_probs]
    
    # Add metadata
    results_df.attrs["processing_time"] = processing_time
    results_df.attrs["avg_time_per_sample"] = processing_time / len(texts)
    
    return results_df

def render_upload_section() -> Optional[pd.DataFrame]:
    """
    Render file upload section.
    
    Returns:
        Uploaded DataFrame or None
    """
    st.markdown("### Data Input")
    
    input_method = st.radio(
        "Select input method:",
        ["File Upload", "Paste CSV", "Sample Data"],
        horizontal=True
    )
    
    df = None
    
    if input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload CSV file:",
            type=['csv'],
            help="CSV should contain a 'text' column with news articles"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
    elif input_method == "Paste CSV":
        csv_text = st.text_area(
            "Paste CSV data (with header):",
            height=200,
            placeholder="text\nYour first news article...\nYour second news article..."
        )
        
        if csv_text:
            try:
                df = pd.read_csv(io.StringIO(csv_text))
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
    
    else:  # Sample Data
        if st.button("Load Sample Data"):
            sample_texts = [
                "UN Security Council discusses global peace initiatives",
                "Lakers win championship in overtime thriller",
                "Tech stocks surge on strong earnings reports",
                "New AI breakthrough in natural language processing",
                "Olympic games set new viewership records",
                "Federal Reserve announces interest rate decision",
                "Climate summit reaches historic agreement",
                "Smartphone sales exceed analyst expectations"
            ]
            df = pd.DataFrame({"text": sample_texts})
    
    # Validate DataFrame
    if df is not None:
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column")
            return None
        
        # Display preview
        st.markdown(f"#### Data Preview ({len(df)} rows)")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Data statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", len(df))
        with col2:
            avg_length = df["text"].str.split().str.len().mean()
            st.metric("Avg Word Count", f"{avg_length:.0f}")
        with col3:
            missing = df["text"].isna().sum()
            st.metric("Missing Values", missing)
    
    return df

def render_results(results_df: pd.DataFrame):
    """
    Render batch processing results.
    
    Args:
        results_df: DataFrame with prediction results
    """
    st.markdown("### Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processed", len(results_df))
    
    with col2:
        avg_confidence = results_df["confidence"].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col3:
        processing_time = results_df.attrs.get("processing_time", 0)
        st.metric("Processing Time", f"{processing_time:.1f}s")
    
    with col4:
        avg_time = results_df.attrs.get("avg_time_per_sample", 0)
        st.metric("Avg Time/Sample", f"{avg_time*1000:.1f}ms")
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Statistics", "Visualization", "Export"])
    
    with tab1:
        # Predictions table
        display_cols = ["text", "predicted_class", "confidence"]
        display_df = results_df[display_cols].copy()
        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
    
    with tab2:
        # Category distribution
        category_counts = results_df["predicted_class"].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Category Distribution")
            for category in AG_NEWS_CLASSES:
                count = category_counts.get(category, 0)
                percentage = count / len(results_df) * 100
                st.metric(category, f"{count} ({percentage:.1f}%)")
        
        with col2:
            st.markdown("#### Confidence Statistics")
            st.metric("Mean", f"{results_df['confidence'].mean():.1%}")
            st.metric("Median", f"{results_df['confidence'].median():.1%}")
            st.metric("Std Dev", f"{results_df['confidence'].std():.1%}")
            st.metric("Min", f"{results_df['confidence'].min():.1%}")
            st.metric("Max", f"{results_df['confidence'].max():.1%}")
    
    with tab3:
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution pie chart
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Prediction Distribution",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence histogram
            fig = px.histogram(
                results_df,
                x="confidence",
                nbins=20,
                title="Confidence Distribution",
                labels={"confidence": "Confidence", "count": "Count"}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence by category
        fig = px.box(
            results_df,
            x="predicted_class",
            y="confidence",
            title="Confidence by Category",
            labels={"predicted_class": "Category", "confidence": "Confidence"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Export options
        st.markdown("#### Export Results")
        
        export_format = st.selectbox(
            "Select format:",
            ["CSV", "JSON", "Excel"]
        )
        
        include_probs = st.checkbox("Include probability scores", value=False)
        
        if include_probs:
            export_df = results_df
        else:
            export_df = results_df[["text", "predicted_class", "confidence"]]
        
        if export_format == "CSV":
            csv = export_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "predictions.csv",
                "text/csv",
                key='download-csv'
            )
        
        elif export_format == "JSON":
            json_str = export_df.to_json(orient='records', indent=2)
            st.download_button(
                "Download JSON",
                json_str,
                "predictions.json",
                "application/json",
                key='download-json'
            )
        
        else:  # Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            st.download_button(
                "Download Excel",
                buffer.getvalue(),
                "predictions.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download-excel'
            )

def render_configuration() -> Dict[str, Any]:
    """Render configuration sidebar."""
    with st.sidebar:
        st.markdown("## Configuration")
        
        # Model settings
        st.markdown("### Model")
        
        model_type = st.selectbox(
            "Model",
            ["DistilBERT", "BERT", "RoBERTa"]
        )
        
        model_map = {
            "DistilBERT": "distilbert-base-uncased",
            "BERT": "bert-base-uncased",
            "RoBERTa": "roberta-base"
        }
        
        # Processing settings
        st.markdown("### Processing")
        
        batch_size = st.slider(
            "Batch Size",
            min_value=8,
            max_value=128,
            value=32,
            step=8,
            help="Larger batch sizes are faster but use more memory"
        )
        
        max_length = st.slider(
            "Max Length",
            min_value=64,
            max_value=512,
            value=256,
            step=32
        )
        
        # Advanced
        with st.expander("Advanced"):
            show_low_confidence = st.checkbox(
                "Highlight low confidence",
                value=True
            )
            
            confidence_threshold = st.slider(
                "Confidence threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.7,
                step=0.05
            )
        
        return {
            "model_path": model_map[model_type],
            "batch_size": batch_size,
            "max_length": max_length,
            "show_low_confidence": show_low_confidence,
            "confidence_threshold": confidence_threshold
        }

def main():
    """Main function for Batch Analysis page."""
    st.set_page_config(
        page_title="Batch Analysis - AG News",
        page_icon="newspaper",
        layout="wide"
    )
    
    st.markdown("# Batch Text Analysis")
    st.markdown("""
    Process multiple news articles simultaneously for efficient classification.
    Upload a CSV file or paste data to classify articles in bulk.
    """)
    
    # Get configuration
    config = render_configuration()
    
    # Upload section
    df = render_upload_section()
    
    # Process button
    if df is not None:
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            process_button = st.button(
                "Process Batch",
                type="primary",
                use_container_width=True
            )
        
        if process_button:
            # Load model
            if "model" not in st.session_state or st.session_state.get("model_path") != config["model_path"]:
                with st.spinner("Loading model..."):
                    model, tokenizer = load_model(config["model_path"])
                    if model and tokenizer:
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.model_path = config["model_path"]
                    else:
                        st.error("Failed to load model")
                        return
            
            # Process batch
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Processing... {progress:.0%}")
            
            with st.spinner("Processing batch..."):
                results_df = process_batch(
                    df["text"].tolist(),
                    st.session_state.model,
                    st.session_state.tokenizer,
                    config["batch_size"],
                    config["max_length"],
                    update_progress
                )
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            render_results(results_df)
            
            # Highlight low confidence if enabled
            if config["show_low_confidence"]:
                low_conf_count = (results_df["confidence"] < config["confidence_threshold"]).sum()
                if low_conf_count > 0:
                    st.warning(f"{low_conf_count} predictions have confidence below {config['confidence_threshold']:.0%}")

if __name__ == "__main__":
    main()
