#!/usr/bin/env python3
"""
Standalone Plotly Dash app for dataset statistics visualization.
Reads from pre-generated CSV files instead of directly processing parquet files.
"""

import argparse
import sys
from pathlib import Path
from typing import List, cast

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.development.base_component import Component
from plotly.subplots import make_subplots

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color"""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def generate_color_variations(base_color, num_colors):
    """Generate color variations from a base color"""
    if num_colors == 1:
        return [base_color]
    
    try:
        # Convert hex to RGB
        base_rgb = hex_to_rgb(base_color)
        
        # Generate variations by adjusting brightness
        colors = []
        for i in range(num_colors):
            # Create variations by blending with white (lighter) and the base color
            factor = 0.3 + (0.7 * i / max(1, num_colors - 1))  # Range from 0.3 to 1.0
            
            new_rgb = tuple(
                int(base_rgb[j] * factor + 255 * (1 - factor))
                for j in range(3)
            )
            colors.append(rgb_to_hex(new_rgb))
        
        return colors
    except Exception:
        # Fallback: return the same color for all
        return [base_color] * num_colors

def load_csv_data(data_dir: Path):
    """Load all CSV data files"""
    data = {}
    
    # Load statistics
    stats_file = data_dir / "statistics.csv"
    if stats_file.exists():
        data["statistics"] = pd.read_csv(stats_file)
    else:
        print(f"Warning: {stats_file} not found")
        data["statistics"] = pd.DataFrame()
    
    # Load split proportions
    split_file = data_dir / "split_proportions.csv"
    if split_file.exists():
        data["split_proportions"] = pd.read_csv(split_file)
    else:
        print(f"Warning: {split_file} not found")
        data["split_proportions"] = pd.DataFrame()
    
    # Load sequence lengths
    seq_file = data_dir / "sequence_lengths.csv"
    if seq_file.exists():
        data["sequence_lengths"] = pd.read_csv(seq_file)
    else:
        print(f"Warning: {seq_file} not found")
        data["sequence_lengths"] = pd.DataFrame()
    
    # Load scores (optional)
    score_file = data_dir / "scores.csv"
    if score_file.exists():
        data["scores"] = pd.read_csv(score_file)
    else:
        data["scores"] = pd.DataFrame()
    
    # Load label counts (optional)
    label_file = data_dir / "label_counts.csv"
    if label_file.exists():
        data["label_counts"] = pd.read_csv(label_file)
    else:
        data["label_counts"] = pd.DataFrame()
    
    return data

def create_pie_charts(split_data):
    """Create pie charts showing split proportions for each dataset"""
    pie_charts = []
    
    if split_data.empty:
        return pie_charts
    
    datasets = split_data["dataset"].unique()
    
    for dataset in datasets:
        dataset_data = split_data[split_data["dataset"] == dataset]
        
        if len(dataset_data) == 0:
            continue
        
        base_color = dataset_data.iloc[0]["color"]
        colors = generate_color_variations(base_color, len(dataset_data))
        
        fig = go.Figure(data=[go.Pie(
            labels=dataset_data["split"],
            values=dataset_data["count"],
            name=dataset,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='inside'
        )])
        
        fig.update_layout(
            title=f"{dataset} Split Distribution",
            showlegend=True,
            width=400,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        pie_charts.append(dcc.Graph(figure=fig))
    
    return pie_charts

def create_histograms(data, column, title, bins=20):
    """Create histograms for the specified column"""
    histograms = []

    if data.empty or column not in data.columns:
        return histograms

    datasets = data["dataset"].unique()

    for dataset in datasets:
        dataset_data = data[data["dataset"] == dataset]
        splits = dataset_data["split"].unique()

        if len(splits) == 0:
            continue

        fig = make_subplots(
            rows=1,
            cols=len(splits),
            subplot_titles=[f"{split.title()}" for split in splits],
            shared_yaxes=True,
            horizontal_spacing=0.08,
        )

        base_color = dataset_data.iloc[0]["color"] if not dataset_data.empty else "#636EFA"
        colors = generate_color_variations(base_color, len(splits))

        for i, split in enumerate(splits):
            split_data = dataset_data[dataset_data["split"] == split]

            if not split_data.empty:
                fig.add_trace(
                    go.Histogram(
                        x=split_data[column],
                        nbinsx=bins,
                        name=f"{dataset} {split}",
                        marker_color=colors[i],
                        showlegend=False,
                    ),
                    row=1,
                    col=i + 1,
                )

        fig.update_layout(
            title=f"{dataset} - {title}",
            showlegend=False,
            width=300 * len(splits),
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )

        histograms.append(dcc.Graph(figure=fig))

    return histograms

def create_boxplots(data, column, title):
    """Create box plots for the specified column"""
    boxplots = []

    if data.empty or column not in data.columns:
        return boxplots

    datasets = data["dataset"].unique()

    for dataset in datasets:
        dataset_data = data[data["dataset"] == dataset]
        splits = dataset_data["split"].unique()

        if len(splits) == 0:
            continue

        fig = go.Figure()
        base_color = dataset_data.iloc[0]["color"] if not dataset_data.empty else "#636EFA"
        colors = generate_color_variations(base_color, len(splits))

        for i, split in enumerate(splits):
            split_data = dataset_data[dataset_data["split"] == split]

            if not split_data.empty:
                fig.add_trace(
                    go.Box(
                        y=split_data[column],
                        name=split,
                        marker_color=colors[i],
                    )
                )

        fig.update_layout(
            title=f"{dataset} - {title}",
            width=400,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )

        boxplots.append(dcc.Graph(figure=fig))

    return boxplots

def create_label_barcharts(label_data):
    """Create bar charts for label counts"""
    barcharts = []

    if label_data.empty:
        return barcharts

    datasets = label_data["dataset"].unique()

    for dataset in datasets:
        dataset_data = label_data[label_data["dataset"] == dataset]
        splits = dataset_data["split"].unique()

        if len(splits) == 0:
            continue

        fig = make_subplots(
            rows=1,
            cols=len(splits),
            subplot_titles=[f"{split.title()}" for split in splits],
            shared_yaxes=True,
            horizontal_spacing=0.08,
        )

        base_color = dataset_data.iloc[0]["color"] if not dataset_data.empty else "#636EFA"
        colors = generate_color_variations(base_color, len(splits))

        for i, split in enumerate(splits):
            split_data = dataset_data[dataset_data["split"] == split]

            if not split_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=split_data["label"],
                        y=split_data["count"],
                        name=f"{dataset} {split}",
                        marker_color=colors[i],
                        showlegend=False,
                    ),
                    row=1,
                    col=i + 1,
                )

        fig.update_layout(
            title=f"{dataset} - Label Counts",
            showlegend=False,
            width=300 * len(splits),
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )

        barcharts.append(dcc.Graph(figure=fig))

    return barcharts

def get_dashboard_components(data):
    stats_df = data["statistics"]
    pie_charts = create_pie_charts(data["split_proportions"])
    seq_histograms = create_histograms(data["sequence_lengths"], "seq_length", "Sequence Length Distribution", 20)
    seq_boxplots = create_boxplots(data["sequence_lengths"], "seq_length", "Sequence Length Distribution")
    score_histograms = create_histograms(data["scores"], "score", "Score Distribution", 20)
    score_boxplots = create_boxplots(data["scores"], "score", "Score Distribution")
    label_barcharts = create_label_barcharts(data["label_counts"])
    return stats_df, pie_charts, seq_histograms, seq_boxplots, score_histograms, score_boxplots, label_barcharts


def create_dashboard_layout(data):
    stats_df, pie_charts, seq_histograms, seq_boxplots, score_histograms, score_boxplots, label_barcharts = get_dashboard_components(data)

    if not stats_df.empty:
        stats_table = dash_table.DataTable(
            data=stats_df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in stats_df.columns],
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
            style_data={"backgroundColor": "rgb(248, 248, 248)"},
            sort_action="native",
            filter_action="native",
        )
    else:
        stats_table = html.Div("No statistics data available")

    layout = html.Div([
        html.H1("Dataset Statistics Dashboard", style={"textAlign": "center", "marginBottom": 30}),
        html.H2("General Statistics"),
        stats_table,
        html.Br(),
    ])

    if pie_charts:
        cast(List[Component], layout.children).extend([
            html.H2("Split Proportions"),
            html.Div(pie_charts, style={"display": "flex", "flexWrap": "wrap"}),
            html.Br(),
        ])

    if seq_histograms:
        cast(List[Component], layout.children).extend([
            html.H2("Sequence Length Distributions"),
            html.Div(seq_histograms, style={"display": "flex", "flexWrap": "wrap"}),
            html.Br(),
        ])

    if seq_boxplots:
        cast(List[Component], layout.children).extend([
            html.H2("Sequence Length Box Plots"),
            html.Div(seq_boxplots, style={"display": "flex", "flexWrap": "wrap"}),
            html.Br(),
        ])

    if score_histograms:
        cast(List[Component], layout.children).extend([
            html.H2("Score Distributions"),
            html.Div(score_histograms, style={"display": "flex", "flexWrap": "wrap"}),
            html.Br(),
        ])

    if score_boxplots:
        cast(List[Component], layout.children).extend([
            html.H2("Score Box Plots"),
            html.Div(score_boxplots, style={"display": "flex", "flexWrap": "wrap"}),
            html.Br(),
        ])

    if label_barcharts:
        cast(List[Component], layout.children).extend([
            html.H2("Label Counts"),
            html.Div(label_barcharts, style={"display": "flex", "flexWrap": "wrap"}),
            html.Br(),
        ])

    return layout

def main():
    """Main function to run the dashboard"""
    parser = argparse.ArgumentParser(description="Run the dataset statistics dashboard")
    parser.add_argument("--data", type=str, required=True, help="Path to the directory containing CSV data files")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind the server to (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    # Validate data directory
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)

    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    # Load data
    print(f"Loading data from {data_dir}...")
    data = load_csv_data(data_dir)

    # Create dashboard
    app = dash.Dash(__name__)
    app.layout = create_dashboard_layout(data)

    # Run server
    print(f"Starting dashboard server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")

if __name__ == "__main__":
    main()
