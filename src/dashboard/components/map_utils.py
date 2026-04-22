"""
components/map_utils.py — Shared choropleth and cluster map helpers.

Owner: Deekshitha (C5)
Phase: 4

Features:
  - Plotly Mapbox choropleth builder
  - DBSCAN cluster scatter overlay
"""

import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np


def create_choropleth_map(data_df: pd.DataFrame, color_column: str, title: str) -> go.Figure:
    """
    Create a Mapbox choropleth map for geographic analysis.
    
    Args:
        data_df: DataFrame with 'latitude', 'longitude', and color_column
        color_column: Column name to color by (e.g., 'call_count', 'avg_response_time')
        title: Map title
    
    Returns:
        Plotly Figure with choropleth map
    """
    
    # Aggregate by geographic area (using lat/lon bins)
    geo_agg = data_df.groupby(
        [pd.cut(data_df['latitude'], bins=20), 
         pd.cut(data_df['longitude'], bins=20)]
    ).agg({
        color_column: 'sum',
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index(drop=True)
    
    fig = go.Figure(data=go.Scattermapbox(
        lat=geo_agg['latitude'],
        lon=geo_agg['longitude'],
        mode='markers',
        marker=dict(
            size=10,
            color=geo_agg[color_column],
            colorscale='Viridis',
            colorbar=dict(title=color_column),
            opacity=0.7,
            line=dict(width=0.5, color='white')
        ),
        hovertemplate='<b>Lat:</b> %{lat:.3f}<br><b>Lon:</b> %{lon:.3f}<br>' +
                     f'<b>{color_column}:</b> %{{marker.color:.0f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        hovermode='closest',
        mapbox=dict(
            accesstoken='pk.eyJ1IjoibWVkYWxlcnRhaSIsImEiOiJjbHQ2dW5pbmwwMDAwMm5wcnF0dWx1bzE4In0.example',  # Replace with actual Mapbox token
            style='open-street-map',
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=data_df['latitude'].mean(),
                lon=data_df['longitude'].mean()
            ),
            pitch=0,
            zoom=10
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def create_cluster_map(data_df: pd.DataFrame, eps: float = 0.05, min_samples: int = 5) -> go.Figure:
    """
    Create a scatter map with DBSCAN clustering overlay.
    
    Args:
        data_df: DataFrame with 'latitude', 'longitude', and optional 'call_count'
        eps: DBSCAN epsilon (neighbor distance in degrees, ~0.05 ≈ 5km)
        min_samples: DBSCAN min_samples parameter
    
    Returns:
        Plotly Figure with cluster map
    """
    
    # Extract coordinates
    coords = data_df[['latitude', 'longitude']].values
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    clusters = clustering.labels_
    
    # Create figure with color coding by cluster
    fig = go.Figure()
    
    # Add clustered points
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            # Noise points (outliers)
            label = 'Noise/Outliers'
            color = 'gray'
            size = 4
        else:
            label = f'Cluster {cluster_id}'
            color = None
            size = 8
        
        mask = clusters == cluster_id
        fig.add_trace(go.Scattermapbox(
            lat=data_df[mask]['latitude'],
            lon=data_df[mask]['longitude'],
            mode='markers',
            name=label,
            marker=dict(
                size=size,
                color=color if color else cluster_id,
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate='<b>Lat:</b> %{lat:.3f}<br><b>Lon:</b> %{lon:.3f}<br>' +
                         f'<b>Cluster:</b> {cluster_id}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Dispatch Hotspot Clusters (DBSCAN)',
        hovermode='closest',
        mapbox=dict(
            style='open-street-map',
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=data_df['latitude'].mean(),
                lon=data_df['longitude'].mean()
            ),
            pitch=0,
            zoom=10
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def create_heatmap(data_df: pd.DataFrame, title: str) -> go.Figure:
    """
    Create a Mapbox heatmap for density visualization.
    
    Args:
        data_df: DataFrame with 'latitude', 'longitude'
        title: Map title
    
    Returns:
        Plotly Figure with heatmap
    """
    
    fig = go.Figure(data=go.Densitymapbox(
        lat=data_df['latitude'],
        lon=data_df['longitude'],
        colorscale='Reds',
        radius=15,
        hovertemplate='<b>Density:</b> %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        mapbox=dict(
            style='open-street-map',
            center=go.layout.mapbox.Center(
                lat=data_df['latitude'].mean(),
                lon=data_df['longitude'].mean()
            ),
            zoom=10
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig
