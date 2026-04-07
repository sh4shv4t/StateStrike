from __future__ import annotations

"""Reusable dashboard components for StateStrike Streamlit UI."""

from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def inject_theme() -> None:
    """Inject custom cyberpunk-inspired CSS theme into Streamlit app."""

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

            :root {
                --bg: #0D0D0F;
                --surface: #16161A;
                --border: #2A2A35;
                --text-primary: #E8E8F0;
                --text-muted: #6B6B85;
                --accent-red: #E8000D;
                --accent-orange: #FF6B00;
                --accent-green: #00E5A0;
                --accent-purple: #9D4EDD;
                --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
            }

            html, body, [class*="css"], .stApp {
                background: radial-gradient(circle at top right, #16161A, #0D0D0F 48%, #09090B 100%);
                color: var(--text-primary);
                font-family: var(--font-mono);
            }

            .block-container {
                padding-top: 1rem;
                max-width: 1320px;
            }

            .muted-header {
                color: var(--text-muted);
                font-size: 0.8rem;
                letter-spacing: 0.06rem;
                text-transform: uppercase;
            }

            .status-dot {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: var(--accent-green);
                margin-right: 8px;
                box-shadow: 0 0 12px var(--accent-green);
                animation: pulseDot 1.2s infinite;
            }

            @keyframes pulseDot {
                0% { opacity: 0.4; transform: scale(1); }
                50% { opacity: 1; transform: scale(1.15); }
                100% { opacity: 0.4; transform: scale(1); }
            }

            .log-panel {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 12px;
                max-height: 460px;
                overflow-y: auto;
                padding: 0.75rem;
            }

            .log-row {
                border-bottom: 1px dashed #242430;
                padding: 0.45rem 0;
                color: var(--text-primary);
                font-size: 0.82rem;
                white-space: pre-wrap;
            }

            .log-row:last-child {
                border-bottom: none;
            }

            .metric-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 0.6rem;
            }

            .badge {
                display: inline-block;
                padding: 0.28rem 0.6rem;
                border-radius: 999px;
                margin-right: 0.45rem;
                font-size: 0.72rem;
                border: 1px solid;
            }

            .badge-on {
                color: #0B0B0F;
                background: var(--accent-green);
                border-color: var(--accent-green);
                box-shadow: 0 0 10px rgba(0, 229, 160, 0.6);
            }

            .badge-off {
                color: var(--text-muted);
                background: #14141A;
                border-color: #2B2B33;
            }

            [data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 0.4rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_action_log(records: list[dict]) -> None:
    """Render latest action rows in a styled scrolling panel.

    Args:
        records: Telemetry records.
    """

    lines: list[str] = []
    for row in records[-20:]:
        status_label = "OK" if int(row.get("http_status", 0)) < 400 else "ERR"
        reward = float(row.get("reward", 0.0))
        reward_text = f"{reward:+.2f} rew"
        line = (
            f"[STEP {int(row.get('step', 0)):03d}] "
            f"{str(row.get('action_type', '')).upper():<10}"
            f" -> {str(row.get('payload_strategy', '')).upper():<9}"
            f" -> {int(row.get('http_status', 0)):3d} {status_label:<3}"
            f" -> {float(row.get('latency_ms', 0.0)):7.1f}ms"
            f" -> {reward_text}"
        )
        lines.append(line)

    html_rows = "".join([f"<div class='log-row'>{line}</div>" for line in lines])
    st.markdown(f"<div class='log-panel'>{html_rows}</div>", unsafe_allow_html=True)


def render_vulnerability_badges(triggered_vulns: list[str]) -> None:
    """Render vulnerability discovery badges.

    Args:
        triggered_vulns: Triggered vulnerability labels.
    """

    redos_on = "redos" in triggered_vulns
    db_on = "db_degradation" in triggered_vulns

    redos_class = "badge badge-on" if redos_on else "badge badge-off"
    db_class = "badge badge-on" if db_on else "badge badge-off"

    redos_label = "REDOS" if redos_on else "REDOS LOCKED"
    db_label = "DB_DEGRADATION" if db_on else "DB_DEGRADATION LOCKED"

    st.markdown(
        (
            f"<span class='{redos_class}'>{redos_label}</span>"
            f"<span class='{db_class}'>{db_label}</span>"
        ),
        unsafe_allow_html=True,
    )


def latency_line_chart(df: pd.DataFrame) -> go.Figure:
    """Build latency line chart with threshold-sensitive color segments.

    Args:
        df: Telemetry dataframe.

    Returns:
        Plotly figure.
    """

    fig = go.Figure()
    if df.empty:
        fig.update_layout(template="plotly_dark", height=260)
        return fig

    x_vals = df["step"]
    y_vals = df["latency_ms"]
    colors = ["#00E5A0" if y < 500 else "#FF6B00" if y < 1500 else "#E8000D" for y in y_vals]

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            marker={"color": colors, "size": 6},
            line={"color": "#00E5A0", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(0, 229, 160, 0.12)",
            name="Latency (ms)",
        )
    )
    fig.update_layout(template="plotly_dark", height=270, margin={"l": 20, "r": 20, "t": 30, "b": 20})
    return fig


def reward_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Build per-step reward bar chart.

    Args:
        df: Telemetry dataframe.

    Returns:
        Plotly figure.
    """

    fig = go.Figure()
    if df.empty:
        fig.update_layout(template="plotly_dark", height=240)
        return fig

    bar_colors = ["#00E5A0" if r >= 0 else "#E8000D" for r in df["reward"]]
    fig.add_trace(
        go.Bar(
            x=df["step"],
            y=df["reward"],
            marker_color=bar_colors,
            name="Reward",
        )
    )
    fig.update_layout(template="plotly_dark", height=250, margin={"l": 20, "r": 20, "t": 25, "b": 20})
    return fig


def action_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Build action distribution donut chart.

    Args:
        df: Telemetry dataframe.

    Returns:
        Plotly figure.
    """

    if df.empty:
        return go.Figure()

    counts = Counter(df["action_type"].tolist())
    fig = px.pie(
        names=list(counts.keys()),
        values=list(counts.values()),
        hole=0.52,
        color_discrete_sequence=["#9D4EDD", "#00E5A0", "#FF6B00", "#E8000D", "#6B6B85"],
    )
    fig.update_layout(template="plotly_dark", height=280, margin={"l": 20, "r": 20, "t": 25, "b": 20})
    return fig


def threat_gauge(cumulative_reward: float) -> go.Figure:
    """Build threat level gauge from cumulative reward.

    Args:
        cumulative_reward: Latest cumulative reward.

    Returns:
        Plotly indicator figure.
    """

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=cumulative_reward,
            title={"text": "THREAT LEVEL"},
            gauge={
                "axis": {"range": [-100, 1500]},
                "bar": {"color": "#9D4EDD"},
                "steps": [
                    {"range": [-100, 100], "color": "#2A2A35"},
                    {"range": [100, 400], "color": "#FF6B00"},
                    {"range": [400, 900], "color": "#E8000D"},
                    {"range": [900, 1500], "color": "#00E5A0"},
                ],
            },
        )
    )
    fig.update_layout(template="plotly_dark", height=260, margin={"l": 20, "r": 20, "t": 40, "b": 20})
    return fig
