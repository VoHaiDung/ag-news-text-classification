"""
Leaderboard Generator for AG News Text Classification
================================================================================
This module generates comprehensive leaderboards from experiment results,
providing rankings, comparisons, and visualizations for model performance.

The generator creates interactive and static leaderboards with support for
multiple metrics, filtering, and export formats.

References:
    - Papers with Code Leaderboards: https://paperswithcode.com/
    - Kaggle Competition Leaderboards: https://www.kaggle.com/

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.io_utils import save_json, load_json
from experiments.results.result_aggregator import ResultAggregator

logger = logging.getLogger(__name__)


class LeaderboardGenerator:
    """
    Generates comprehensive leaderboards from experiment results.
    
    This class provides:
    - Multi-metric ranking
    - Interactive visualizations
    - Export to multiple formats
    - Historical tracking
    - Comparison tools
    """
    
    def __init__(
        self,
        results_dir: str = "outputs/results",
        output_dir: str = "outputs/leaderboards",
        primary_metric: str = "accuracy",
        secondary_metrics: Optional[List[str]] = None
    ):
        """
        Initialize leaderboard generator.
        
        Args:
            results_dir: Directory containing experiment results
            output_dir: Directory to save leaderboards
            primary_metric: Primary metric for ranking
            secondary_metrics: Additional metrics to display
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.primary_metric = primary_metric
        self.secondary_metrics = secondary_metrics or [
            "f1_score", "precision", "recall", "inference_speed", "model_size"
        ]
        
        self.result_aggregator = ResultAggregator(results_dir)
        self.leaderboard_data = None
        self.rankings = {}
        
        logger.info(f"Initialized LeaderboardGenerator with primary metric: {primary_metric}")
    
    def generate_leaderboard(
        self,
        experiment_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_ensemble: bool = True
    ) -> pd.DataFrame:
        """
        Generate leaderboard from experiments.
        
        Args:
            experiment_ids: Specific experiments to include
            filters: Filters to apply
            include_ensemble: Whether to include ensemble models
            
        Returns:
            Leaderboard DataFrame
        """
        # Load experiments
        experiments = self.result_aggregator.load_experiments(experiment_ids)
        
        # Prepare leaderboard data
        leaderboard_entries = []
        
        for exp_id, exp_data in experiments.items():
            entry = self._create_leaderboard_entry(exp_id, exp_data)
            
            # Apply filters
            if filters and not self._apply_filters(entry, filters):
                continue
            
            # Check ensemble filter
            if not include_ensemble and entry.get("is_ensemble", False):
                continue
            
            leaderboard_entries.append(entry)
        
        # Create DataFrame
        self.leaderboard_data = pd.DataFrame(leaderboard_entries)
        
        if self.leaderboard_data.empty:
            logger.warning("No experiments found for leaderboard")
            return self.leaderboard_data
        
        # Calculate rankings
        self._calculate_rankings()
        
        # Sort by primary metric
        self.leaderboard_data = self.leaderboard_data.sort_values(
            self.primary_metric,
            ascending=False
        )
        
        # Add rank column
        self.leaderboard_data["rank"] = range(1, len(self.leaderboard_data) + 1)
        
        logger.info(f"Generated leaderboard with {len(self.leaderboard_data)} entries")
        return self.leaderboard_data
    
    def _create_leaderboard_entry(
        self,
        exp_id: str,
        exp_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create leaderboard entry from experiment data.
        
        Args:
            exp_id: Experiment ID
            exp_data: Experiment data
            
        Returns:
            Leaderboard entry
        """
        entry = {
            "experiment_id": exp_id,
            "model_name": exp_data.get("config", {}).get("model_name", "Unknown"),
            "timestamp": exp_data.get("timestamp"),
            "is_ensemble": exp_data.get("config", {}).get("is_ensemble", False)
        }
        
        # Add primary metric
        primary_value = self._extract_metric_value(exp_data, self.primary_metric)
        entry[self.primary_metric] = primary_value
        
        # Add secondary metrics
        for metric in self.secondary_metrics:
            value = self._extract_metric_value(exp_data, metric)
            entry[metric] = value
        
        # Add configuration details
        config = exp_data.get("config", {})
        entry["learning_rate"] = config.get("learning_rate")
        entry["batch_size"] = config.get("batch_size")
        entry["num_epochs"] = config.get("num_epochs")
        entry["optimizer"] = config.get("optimizer")
        
        # Add resource usage
        entry["training_time_hours"] = exp_data.get("training_time", 0) / 3600
        entry["gpu_memory_gb"] = exp_data.get("gpu_memory", 0) / 1024
        
        # Calculate efficiency score
        entry["efficiency_score"] = self._calculate_efficiency_score(entry)
        
        return entry
    
    def _extract_metric_value(
        self,
        exp_data: Dict[str, Any],
        metric: str
    ) -> Optional[float]:
        """
        Extract metric value from experiment data.
        
        Args:
            exp_data: Experiment data
            metric: Metric name
            
        Returns:
            Metric value
        """
        # Try different locations
        locations = [
            exp_data.get("metrics", {}),
            exp_data.get("results", {}),
            exp_data.get("results", {}).get("metrics", {}),
            exp_data.get("results", {}).get("test", {})
        ]
        
        for location in locations:
            if metric in location:
                value = location[metric]
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, dict):
                    # Try to get mean or value
                    if "mean" in value:
                        return float(value["mean"])
                    elif "value" in value:
                        return float(value["value"])
        
        return None
    
    def _calculate_efficiency_score(
        self,
        entry: Dict[str, Any]
    ) -> float:
        """
        Calculate efficiency score combining performance and resource usage.
        
        Args:
            entry: Leaderboard entry
            
        Returns:
            Efficiency score
        """
        # Normalize metrics
        performance = entry.get(self.primary_metric, 0)
        
        # Inverse metrics (lower is better)
        training_time = entry.get("training_time_hours", 1)
        gpu_memory = entry.get("gpu_memory_gb", 1)
        model_size = entry.get("model_size", 100)
        
        # Calculate efficiency score
        # Higher performance with lower resource usage is better
        efficiency = performance / (
            np.log1p(training_time) * np.log1p(gpu_memory) * np.log1p(model_size / 100)
        )
        
        return float(efficiency)
    
    def _apply_filters(
        self,
        entry: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """
        Apply filters to leaderboard entry.
        
        Args:
            entry: Leaderboard entry
            filters: Filters to apply
            
        Returns:
            True if entry passes filters
        """
        for key, value in filters.items():
            if key not in entry:
                return False
            
            if isinstance(value, (list, tuple)):
                if entry[key] not in value:
                    return False
            elif isinstance(value, dict):
                # Range filter
                if "min" in value and entry[key] < value["min"]:
                    return False
                if "max" in value and entry[key] > value["max"]:
                    return False
            else:
                if entry[key] != value:
                    return False
        
        return True
    
    def _calculate_rankings(self):
        """Calculate rankings for all metrics."""
        self.rankings = {}
        
        metrics = [self.primary_metric] + self.secondary_metrics
        
        for metric in metrics:
            if metric in self.leaderboard_data.columns:
                # Handle NaN values
                valid_data = self.leaderboard_data[metric].dropna()
                
                if not valid_data.empty:
                    # Rank (higher is better for most metrics)
                    if metric in ["training_time_hours", "gpu_memory_gb", "model_size"]:
                        # Lower is better
                        ranks = valid_data.rank(ascending=True)
                    else:
                        # Higher is better
                        ranks = valid_data.rank(ascending=False)
                    
                    self.rankings[metric] = ranks.to_dict()
    
    def create_visualizations(
        self,
        save_static: bool = True,
        save_interactive: bool = True
    ):
        """
        Create leaderboard visualizations.
        
        Args:
            save_static: Save static plots
            save_interactive: Save interactive plots
        """
        if self.leaderboard_data is None or self.leaderboard_data.empty:
            logger.warning("No leaderboard data to visualize")
            return
        
        if save_static:
            self._create_static_visualizations()
        
        if save_interactive:
            self._create_interactive_visualizations()
    
    def _create_static_visualizations(self):
        """Create static matplotlib visualizations."""
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Top models bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        top_n = min(10, len(self.leaderboard_data))
        top_models = self.leaderboard_data.head(top_n)
        
        bars = ax.bar(
            range(top_n),
            top_models[self.primary_metric],
            color='steelblue'
        )
        
        # Add value labels
        for i, (idx, row) in enumerate(top_models.iterrows()):
            ax.text(
                i, row[self.primary_metric] + 0.001,
                f"{row[self.primary_metric]:.4f}",
                ha='center', va='bottom'
            )
        
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(top_models["model_name"], rotation=45, ha='right')
        ax.set_xlabel("Model")
        ax.set_ylabel(self.primary_metric.capitalize())
        ax.set_title(f"Top {top_n} Models by {self.primary_metric.capitalize()}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "top_models.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Multi-metric comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics_to_plot = [
            self.primary_metric,
            "f1_score",
            "efficiency_score",
            "training_time_hours"
        ]
        
        for idx, metric in enumerate(metrics_to_plot):
            if metric in self.leaderboard_data.columns:
                ax = axes[idx // 2, idx % 2]
                
                top_models = self.leaderboard_data.head(10)
                
                if metric == "training_time_hours":
                    # Lower is better
                    colors = ['red' if v > top_models[metric].median() else 'green' 
                             for v in top_models[metric]]
                else:
                    # Higher is better
                    colors = ['green' if v > top_models[metric].median() else 'red' 
                             for v in top_models[metric]]
                
                ax.barh(range(len(top_models)), top_models[metric], color=colors)
                ax.set_yticks(range(len(top_models)))
                ax.set_yticklabels(top_models["model_name"])
                ax.set_xlabel(metric.replace("_", " ").capitalize())
                ax.set_title(f"Model Comparison: {metric.replace('_', ' ').capitalize()}")
                ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "multi_metric_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Scatter plot: Performance vs Efficiency
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if "efficiency_score" in self.leaderboard_data.columns:
            scatter = ax.scatter(
                self.leaderboard_data["efficiency_score"],
                self.leaderboard_data[self.primary_metric],
                s=100,
                c=self.leaderboard_data["rank"],
                cmap='viridis_r',
                alpha=0.6
            )
            
            # Add model labels for top models
            for idx, row in self.leaderboard_data.head(5).iterrows():
                ax.annotate(
                    row["model_name"],
                    (row["efficiency_score"], row[self.primary_metric]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
            
            ax.set_xlabel("Efficiency Score")
            ax.set_ylabel(self.primary_metric.capitalize())
            ax.set_title("Performance vs Efficiency Trade-off")
            
            plt.colorbar(scatter, label="Rank")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_efficiency.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_interactive_visualizations(self):
        """Create interactive Plotly visualizations."""
        # 1. Interactive leaderboard table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(self.leaderboard_data.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[self.leaderboard_data[col] for col in self.leaderboard_data.columns],
                fill_color='lavender',
                align='left',
                format=[".4f" if self.leaderboard_data[col].dtype in [np.float64, np.float32] else None 
                       for col in self.leaderboard_data.columns]
            )
        )])
        
        fig.update_layout(
            title="Interactive Leaderboard",
            height=600
        )
        
        fig.write_html(self.output_dir / "interactive_leaderboard.html")
        
        # 2. Parallel coordinates plot
        numeric_cols = self.leaderboard_data.select_dtypes(include=[np.number]).columns
        plot_cols = [col for col in numeric_cols if col != "rank"][:6]  # Limit to 6 metrics
        
        if plot_cols:
            fig = px.parallel_coordinates(
                self.leaderboard_data.head(20),
                dimensions=plot_cols,
                color="rank",
                color_continuous_scale=px.colors.diverging.Tealrose,
                labels={col: col.replace("_", " ").title() for col in plot_cols},
                title="Multi-Metric Model Comparison"
            )
            
            fig.write_html(self.output_dir / "parallel_coordinates.html")
        
        # 3. Radar chart for top models
        top_models = self.leaderboard_data.head(5)
        
        # Select metrics for radar chart
        radar_metrics = [self.primary_metric] + [
            m for m in ["f1_score", "precision", "recall", "efficiency_score"] 
            if m in self.leaderboard_data.columns
        ]
        
        fig = go.Figure()
        
        for idx, row in top_models.iterrows():
            values = [row[metric] for metric in radar_metrics]
            
            # Normalize values to 0-1 range for better visualization
            normalized_values = []
            for metric, value in zip(radar_metrics, values):
                col_min = self.leaderboard_data[metric].min()
                col_max = self.leaderboard_data[metric].max()
                if col_max > col_min:
                    normalized_values.append((value - col_min) / (col_max - col_min))
                else:
                    normalized_values.append(0.5)
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=radar_metrics,
                fill='toself',
                name=row["model_name"]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Top Models Comparison (Normalized Metrics)"
        )
        
        fig.write_html(self.output_dir / "radar_chart.html")
    
    def export_leaderboard(
        self,
        formats: List[str] = ["csv", "json", "html", "latex"]
    ):
        """
        Export leaderboard to various formats.
        
        Args:
            formats: List of export formats
        """
        if self.leaderboard_data is None or self.leaderboard_data.empty:
            logger.warning("No leaderboard data to export")
            return
        
        for fmt in formats:
            if fmt == "csv":
                output_path = self.output_dir / "leaderboard.csv"
                self.leaderboard_data.to_csv(output_path, index=False)
                logger.info(f"Exported to {output_path}")
            
            elif fmt == "json":
                output_path = self.output_dir / "leaderboard.json"
                self.leaderboard_data.to_json(output_path, orient="records", indent=2)
                logger.info(f"Exported to {output_path}")
            
            elif fmt == "html":
                output_path = self.output_dir / "leaderboard.html"
                html = self._generate_html_leaderboard()
                with open(output_path, "w") as f:
                    f.write(html)
                logger.info(f"Exported to {output_path}")
            
            elif fmt == "latex":
                output_path = self.output_dir / "leaderboard.tex"
                latex = self.leaderboard_data.to_latex(index=False, float_format="%.4f")
                with open(output_path, "w") as f:
                    f.write(latex)
                logger.info(f"Exported to {output_path}")
            
            elif fmt == "markdown":
                output_path = self.output_dir / "leaderboard.md"
                markdown = self._generate_markdown_leaderboard()
                with open(output_path, "w") as f:
                    f.write(markdown)
                logger.info(f"Exported to {output_path}")
    
    def _generate_html_leaderboard(self) -> str:
        """
        Generate HTML leaderboard with styling.
        
        Returns:
            HTML string
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AG News Classification Leaderboard</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                .metadata {
                    text-align: center;
                    color: #666;
                    margin-bottom: 20px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                th {
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    position: sticky;
                    top: 0;
                }
                td {
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }
                tr:hover {
                    background-color: #f1f1f1;
                }
                .rank-1 { background-color: #ffd700; }
                .rank-2 { background-color: #c0c0c0; }
                .rank-3 { background-color: #cd7f32; }
            </style>
        </head>
        <body>
            <h1>AG News Text Classification Leaderboard</h1>
            <div class="metadata">
                <p>Generated: {timestamp}</p>
                <p>Total Models: {num_models}</p>
                <p>Primary Metric: {primary_metric}</p>
            </div>
            {table_html}
        </body>
        </html>
        """
        
        # Convert DataFrame to HTML table
        table_html = self.leaderboard_data.to_html(
            index=False,
            float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "",
            classes="leaderboard-table",
            table_id="leaderboard"
        )
        
        # Add row highlighting for top 3
        table_html = table_html.replace('<tr>', '<tr class="rank-{}">')
        
        return html.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            num_models=len(self.leaderboard_data),
            primary_metric=self.primary_metric,
            table_html=table_html
        )
    
    def _generate_markdown_leaderboard(self) -> str:
        """
        Generate Markdown leaderboard.
        
        Returns:
            Markdown string
        """
        md = f"""# AG News Text Classification Leaderboard

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Total Models:** {len(self.leaderboard_data)}  
**Primary Metric:** {self.primary_metric}

## Top Performers

"""
        
        # Add table
        md += self.leaderboard_data.head(20).to_markdown(index=False, floatfmt=".4f")
        
        # Add summary statistics
        md += f"""

## Summary Statistics

- **Best {self.primary_metric}:** {self.leaderboard_data[self.primary_metric].max():.4f}
- **Average {self.primary_metric}:** {self.leaderboard_data[self.primary_metric].mean():.4f}
- **Median {self.primary_metric}:** {self.leaderboard_data[self.primary_metric].median():.4f}
"""
        
        # Add best model details
        best_model = self.leaderboard_data.iloc[0]
        md += f"""

## Best Model Details

**Model:** {best_model['model_name']}  
**{self.primary_metric}:** {best_model[self.primary_metric]:.4f}  
"""
        
        for metric in self.secondary_metrics:
            if metric in best_model and pd.notna(best_model[metric]):
                md += f"**{metric}:** {best_model[metric]:.4f}  \n"
        
        return md
    
    def update_historical_leaderboard(self):
        """Update historical leaderboard tracking."""
        historical_file = self.output_dir / "historical_leaderboard.json"
        
        # Load existing historical data
        if historical_file.exists():
            historical_data = load_json(historical_file)
        else:
            historical_data = []
        
        # Add current snapshot
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "num_models": len(self.leaderboard_data),
            "best_model": self.leaderboard_data.iloc[0]["model_name"] if not self.leaderboard_data.empty else None,
            "best_score": float(self.leaderboard_data.iloc[0][self.primary_metric]) if not self.leaderboard_data.empty else None,
            "top_5": self.leaderboard_data.head(5)[["model_name", self.primary_metric]].to_dict("records")
        }
        
        historical_data.append(snapshot)
        
        # Save updated historical data
        save_json(historical_data, historical_file)
        
        logger.info(f"Updated historical leaderboard with {len(historical_data)} snapshots")
    
    def compare_models(
        self,
        model_names: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare specific models.
        
        Args:
            model_names: List of model names to compare
            metrics: Metrics to compare (uses all if None)
            
        Returns:
            Comparison DataFrame
        """
        if self.leaderboard_data is None or self.leaderboard_data.empty:
            logger.warning("No leaderboard data available")
            return pd.DataFrame()
        
        # Filter models
        comparison_data = self.leaderboard_data[
            self.leaderboard_data["model_name"].isin(model_names)
        ]
        
        if comparison_data.empty:
            logger.warning(f"No models found matching: {model_names}")
            return comparison_data
        
        # Select metrics
        if metrics:
            cols = ["model_name", "rank"] + metrics
            cols = [c for c in cols if c in comparison_data.columns]
            comparison_data = comparison_data[cols]
        
        return comparison_data
