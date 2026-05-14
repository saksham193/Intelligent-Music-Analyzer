"""3D music galaxy visualization."""

from __future__ import annotations

from typing import Dict, List

import plotly.express as px


def music_galaxy_figure(points: List[Dict[str, object]]):
    """Create a 3D scatter where x=energy, y=emotion/valence, z=tempo."""
    if not points:
        points = [
            {
                "title": "Current song",
                "energy": 0.5,
                "valence": 0.5,
                "tempo": 100.0,
                "cluster": "unknown",
            }
        ]

    fig = px.scatter_3d(
        points,
        x="energy",
        y="valence",
        z="tempo",
        color="cluster",
        hover_name="title",
        labels={"energy": "Energy", "valence": "Emotion / Valence", "tempo": "Tempo"},
    )
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0))
    return fig
