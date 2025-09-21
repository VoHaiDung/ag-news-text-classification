"""
Performance Dashboard Page for AG News Classification
======================================================

Implements performance monitoring dashboard following principles from:
- Few (2006): "Information Dashboard Design" - Visual display of information
- Tufte (2001): "The Visual Display of Quantitative Information"
- Cairo (2016): "The Truthful Art" - Data visualization and communication

Author: Võ Hải Dũng
License: MIT
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

# Add project root to path
PAGES_DIR = Path(__file__).parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from app import get_app_controller
from configs.constants import AG_NEWS_CLASSES
from src.utils.logging_config import setup_logging
from src.utils.experiment_tracking import ExperimentTracker

logger = setup_logging(__name__)

def load_performance_metrics() -> Dict[str, Any]:
    """
    Load performance metrics from tracking system.
    
    Returns:
        Dictionary with performance data
    """
    # Simulated data - in production, load from MLflow/Wandb
    np.random.seed(42)
    
    # Generate time series data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    metrics = {
        "accuracy": {
            "current": 0.962,
            "history": 0.95 + np.random.randn(30) * 0.01 + np.linspace(0, 0.012, 30),
            "dates": dates
        },
        "f1_score": {
            "current": 0.958,
            "history": 0.94 + np.random.randn(30) * 0.01 + np.linspace(0, 0.018, 30),
            "dates": dates
        },
        "latency": {
            "current": 45,
            "history": 50 - np.random.randn(30) * 5 - np.linspace(0, 5, 30),
            "dates": dates,
            "unit": "ms"
        },
        "throughput": {
            "current": 1234,
            "history": 1000 + np.random.randn(30) * 100 + np.linspace(0, 234, 30),
            "dates": dates,
            "unit": "req/min"
        },
        "class_metrics": {
            class_name: {
                "precision": 0.95 + np.random.random() * 0.03,
                "recall": 0.94 + np.random.random() * 0.04,
                "f1": 0.945 + np.random.random() * 0.035,
                "support": np.random.randint(2000, 3000)
            }
            for class_name in AG_NEWS_CLASSES
        },
        "confusion_matrix": np.random.randint(0, 50, (4, 4)) + np.eye(4) * 450,
        "resource_usage": {
            "cpu": 45 + np.random.randn(30) * 10,
            "memory": 2.8 + np.random.randn(30) * 0.3,
            "gpu": 65 + np.random.randn(30) * 15,
            "dates": dates
        }
    }
    
    return metrics

def render_metrics_overview(metrics: Dict[str, Any]):
    """
    Render metrics overview section.
    
    Args:
        metrics: Performance metrics dictionary
    """
    st.markdown("### System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta = metrics["accuracy"]["current"] - metrics["accuracy"]["history"][-2]
        st.metric(
            "Accuracy",
            f"{metrics['accuracy']['current']:.1%}",
            f"{delta:+.2%}"
        )
    
    with col2:
        delta = metrics["f1_score"]["current"] - metrics["f1_score"]["history"][-2]
        st.metric(
            "F1 Score",
            f"{metrics['f1_score']['current']:.1%}",
            f"{delta:+.2%}"
        )
    
    with col3:
        delta = metrics["latency"]["current"] - metrics["latency"]["history"][-2]
        st.metric(
            "Avg Latency",
            f"{metrics['latency']['current']:.0f} ms",
            f"{delta:+.1f} ms",
            delta_color="inverse"
        )
    
    with col4:
        delta = metrics["throughput"]["current"] - metrics["throughput"]["history"][-2]
        st.metric(
            "Throughput",
            f"{metrics['throughput']['current']:.0f} req/min",
            f"{delta:+.0f}"
        )

def render_time_series_charts(metrics: Dict[str, Any]):
    """
    Render time series performance charts.
    
    Args:
        metrics: Performance metrics dictionary
    """
    st.markdown("### Performance Trends")
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Accuracy", "F1 Score", "Latency", "Throughput"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Accuracy trend
    fig.add_trace(
        go.Scatter(
            x=metrics["accuracy"]["dates"],
            y=metrics["accuracy"]["history"],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # F1 Score trend
    fig.add_trace(
        go.Scatter(
            x=metrics["f1_score"]["dates"],
            y=metrics["f1_score"]["history"],
            mode='lines+markers',
            name='F1 Score',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ),
        row=1, col=2
    )
    
    # Latency trend
    fig.add_trace(
        go.Scatter(
            x=metrics["latency"]["dates"],
            y=metrics["latency"]["history"],
            mode='lines+markers',
            name='Latency',
            line=dict(color='orange', width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    # Throughput trend
    fig.add_trace(
        go.Scatter(
            x=metrics["throughput"]["dates"],
            y=metrics["throughput"]["history"],
            mode='lines+markers',
            name='Throughput',
            line=dict(color='purple', width=2),
            marker=dict(size=4)
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="30-Day Performance Trends"
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=2)
    fig.update_yaxes(title_text="ms", row=2, col=1)
    fig.update_yaxes(title_text="req/min", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def render_class_performance(metrics: Dict[str, Any]):
    """
    Render per-class performance metrics.
    
    Args:
        metrics: Performance metrics dictionary
    """
    st.markdown("### Per-Class Performance")
    
    # Prepare data
    class_data = []
    for class_name, class_metrics in metrics["class_metrics"].items():
        class_data.append({
            "Class": class_name,
            "Precision": class_metrics["precision"],
            "Recall": class_metrics["recall"],
            "F1 Score": class_metrics["f1"],
            "Support": class_metrics["support"]
        })
    
    df = pd.DataFrame(class_data)
    
    # Create grouped bar chart
    fig = go.Figure()
    
    for metric in ["Precision", "Recall", "F1 Score"]:
        fig.add_trace(go.Bar(
            name=metric,
            x=df["Class"],
            y=df[metric],
            text=[f"{v:.1%}" for v in df[metric]],
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Classification Metrics by Category",
        xaxis_title="Category",
        yaxis_title="Score",
        barmode='group',
        height=400,
        yaxis_range=[0.9, 1.0]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    with st.expander("Detailed Metrics"):
        detailed_df = df.copy()
        for col in ["Precision", "Recall", "F1 Score"]:
            detailed_df[col] = detailed_df[col].apply(lambda x: f"{x:.3%}")
        
        st.dataframe(detailed_df, use_container_width=True, hide_index=True)

def render_confusion_matrix(metrics: Dict[str, Any]):
    """
    Render confusion matrix.
    
    Args:
        metrics: Performance metrics dictionary
    """
    st.markdown("### Confusion Matrix")
    
    cm = metrics["confusion_matrix"]
    
    # Normalize for display
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=AG_NEWS_CLASSES,
        y=AG_NEWS_CLASSES,
        text=cm.astype(int),
        texttemplate="%{text}",
        colorscale='Blues',
        colorbar=dict(title="Proportion")
    ))
    
    fig.update_layout(
        title="Confusion Matrix (Last 2000 Predictions)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_resource_usage(metrics: Dict[str, Any]):
    """
    Render resource usage metrics.
    
    Args:
        metrics: Performance metrics dictionary
    """
    st.markdown("### Resource Usage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CPU usage gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=metrics["resource_usage"]["cpu"][-1],
            title={'text': "CPU Usage (%)"},
            delta={'reference': metrics["resource_usage"]["cpu"][-2]},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Memory usage gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=metrics["resource_usage"]["memory"][-1],
            title={'text': "Memory (GB)"},
            delta={'reference': metrics["resource_usage"]["memory"][-2]},
            gauge={'axis': {'range': [None, 8]},
                   'bar': {'color': "darkgreen"},
                   'steps': [
                       {'range': [0, 4], 'color': "lightgray"},
                       {'range': [4, 6], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 7}}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # GPU usage gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=metrics["resource_usage"]["gpu"][-1],
            title={'text': "GPU Usage (%)"},
            delta={'reference': metrics["resource_usage"]["gpu"][-2]},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkorange"},
                   'steps': [
                       {'range': [0, 60], 'color': "lightgray"},
                       {'range': [60, 85], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 95}}
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

def render_configuration() -> Dict[str, Any]:
    """Render configuration sidebar."""
    with st.sidebar:
        st.markdown("## Dashboard Settings")
        
        # Time range
        st.markdown("### Time Range")
        
        time_range = st.selectbox(
            "Select Period",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
        )
        
        # Refresh settings
        st.markdown("### Refresh")
        
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        
        if auto_refresh:
            refresh_interval = st.slider(
                "Interval (seconds)",
                min_value=5,
                max_value=60,
                value=30
            )
        else:
            refresh_interval = None
        
        # Display settings
        st.markdown("### Display")
        
        show_trends = st.checkbox("Show Trends", value=True)
        show_confusion = st.checkbox("Show Confusion Matrix", value=True)
        show_resources = st.checkbox("Show Resource Usage", value=True)
        
        # Export settings
        with st.expander("Export Options"):
            export_format = st.selectbox(
                "Format",
                ["JSON", "CSV", "PDF Report"]
            )
            
            include_raw_data = st.checkbox("Include Raw Data", value=False)
        
        return {
            "time_range": time_range,
            "auto_refresh": auto_refresh,
            "refresh_interval": refresh_interval,
            "show_trends": show_trends,
            "show_confusion": show_confusion,
            "show_resources": show_resources,
            "export_format": export_format,
            "include_raw_data": include_raw_data
        }

def main():
    """Main function for Performance Dashboard page."""
    st.set_page_config(
        page_title="Performance Dashboard - AG News",
        page_icon="📊",
        layout="wide"
    )
    
    st.markdown("# Performance Dashboard")
    st.markdown("""
    Real-time monitoring of model performance, system metrics, and resource utilization.
    """)
    
    # Get configuration
    config = render_configuration()
    
    # Load metrics
    metrics = load_performance_metrics()
    
    # Auto-refresh
    if config["auto_refresh"] and config["refresh_interval"]:
        st.empty()
        import time
        time.sleep(config["refresh_interval"])
        st.experimental_rerun()
    
    # Render sections
    render_metrics_overview(metrics)
    
    st.markdown("---")
    
    if config["show_trends"]:
        render_time_series_charts(metrics)
        st.markdown("---")
    
    # Two columns for class performance and confusion matrix
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_class_performance(metrics)
    
    with col2:
        if config["show_confusion"]:
            render_confusion_matrix(metrics)
    
    if config["show_resources"]:
        st.markdown("---")
        render_resource_usage(metrics)
    
    # Export functionality
    st.markdown("---")
    st.markdown("### Export Dashboard")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("Export Metrics", type="primary"):
            if config["export_format"] == "JSON":
                # Prepare export data
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "accuracy": metrics["accuracy"]["current"],
                    "f1_score": metrics["f1_score"]["current"],
                    "latency": metrics["latency"]["current"],
                    "throughput": metrics["throughput"]["current"]
                }
                
                if config["include_raw_data"]:
                    export_data["raw_metrics"] = {
                        k: v for k, v in metrics.items()
                        if k != "confusion_matrix"
                    }
                
                st.download_button(
                    "Download JSON",
                    json.dumps(export_data, indent=2, default=str),
                    "performance_metrics.json",
                    "application/json"
                )
            
            elif config["export_format"] == "CSV":
                # Create DataFrame
                df = pd.DataFrame({
                    "Metric": ["Accuracy", "F1 Score", "Latency", "Throughput"],
                    "Value": [
                        metrics["accuracy"]["current"],
                        metrics["f1_score"]["current"],
                        metrics["latency"]["current"],
                        metrics["throughput"]["current"]
                    ]
                })
                
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False),
                    "performance_metrics.csv",
                    "text/csv"
                )

if __name__ == "__main__":
    main()
