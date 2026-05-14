"""Advanced analytics charts for the dashboard."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def daily_mood_timeline(history: List[Dict[str, object]]) -> go.Figure:
    """Plot mood sessions over time."""
    if not history:
        return px.line(x=[], y=[], title="Daily Mood Timeline")
    df = pd.DataFrame(history)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    counts = df.groupby([df["created_at"].dt.date, "final_emotion"]).size().reset_index(name="sessions")
    fig = px.line(counts, x="created_at", y="sessions", color="final_emotion", markers=True)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def language_preference_chart(history: List[Dict[str, object]]) -> go.Figure:
    languages = [str(item.get("language", "unknown")) for item in history] or ["unknown"]
    counts = Counter(languages)
    fig = px.pie(values=list(counts.values()), names=list(counts.keys()), hole=0.45)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def recommendation_feedback_chart(feedback: List[Dict[str, object]]) -> go.Figure:
    actions = [str(item.get("action", "none")) for item in feedback] or ["none"]
    fig = px.histogram(x=actions, labels={"x": "Feedback", "y": "Count"})
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def mood_heatmap(history: List[Dict[str, object]]) -> go.Figure:
    if not history:
        return px.imshow([[0]], labels=dict(x="Hour", y="Mood", color="Sessions"))
    df = pd.DataFrame(history)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["hour"] = df["created_at"].dt.hour.fillna(0).astype(int)
    pivot = pd.crosstab(df["final_emotion"], df["hour"])
    fig = px.imshow(pivot, labels=dict(x="Hour", y="Mood", color="Sessions"), aspect="auto")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig

