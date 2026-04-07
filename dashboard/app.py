from __future__ import annotations

"""Streamlit real-time dashboard for StateStrike telemetry."""

import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.telemetry import TelemetryWriter
from dashboard.components import (
    action_donut_chart,
    inject_theme,
    latency_line_chart,
    render_action_log,
    render_vulnerability_badges,
    reward_bar_chart,
    threat_gauge,
)

load_dotenv()

TELEMETRY_FILE = os.getenv("TELEMETRY_FILE", "telemetry.json")

st.set_page_config(page_title="StateStrike Dashboard", layout="wide")
inject_theme()

st.markdown("<div class='muted-header'>StateStrike | OpenEnv Hackathon 2025 | Meta × Hugging Face</div>", unsafe_allow_html=True)
st.markdown("## <span class='status-dot'></span>STATESTRIKE AGENT", unsafe_allow_html=True)

records = TelemetryWriter.read_recent(n=300, file_path=TELEMETRY_FILE)
df = pd.DataFrame(records)
last = records[-1] if records else {}

step = int(last.get("step", 0)) if last else 0
cumulative_reward = float(last.get("cumulative_reward", 0.0)) if last else 0.0
triggered_vulns = list(last.get("triggered_vulns", [])) if last else []
avg_latency = float(df["latency_ms"].mean()) if not df.empty else 0.0


def classify_threat_level(score: float) -> str:
    """Map cumulative reward to a threat-tier label.

    Args:
        score: Cumulative reward.

    Returns:
        Threat level string.
    """

    if score < 100:
        return "NONE"
    if score < 400:
        return "ELEVATED"
    if score < 900:
        return "CRITICAL"
    return "SYSTEM BREACH"


threat_level = classify_threat_level(cumulative_reward)

metric_columns = st.columns(4)
metric_columns[0].metric("Step", step)
metric_columns[1].metric("Cumulative Reward", f"{cumulative_reward:.2f}")
metric_columns[2].metric("Vulns Found", len(triggered_vulns))
metric_columns[3].metric("Avg Latency", f"{avg_latency:.1f} ms")

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.subheader("Agent Activity Feed")
    render_action_log(records)

    st.markdown("### Reward Breakdown")
    breakdown = last.get("reward_breakdown", {}) if isinstance(last, dict) else {}
    breakdown_df = pd.DataFrame(
        [
            {"Component": "alpha*log latency", "Value": float(breakdown.get("latency_reward", 0.0))},
            {"Component": "beta chain bonus", "Value": float(breakdown.get("chain_bonus", 0.0))},
            {"Component": "gamma exploit bounty", "Value": float(breakdown.get("exploit_bounty", 0.0))},
            {"Component": "delta fuzz penalty", "Value": float(breakdown.get("fuzz_penalty", 0.0))},
            {"Component": "total", "Value": float(breakdown.get("total", 0.0))},
        ]
    )
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    st.markdown("### Triggered Vulnerabilities")
    render_vulnerability_badges(triggered_vulns)

with right_col:
    st.subheader("Live Metrics")
    st.plotly_chart(latency_line_chart(df.tail(100)), use_container_width=True)
    st.plotly_chart(reward_bar_chart(df.tail(100)), use_container_width=True)
    st.plotly_chart(action_donut_chart(df.tail(100)), use_container_width=True)
    st.markdown(f"**Threat Classification:** {threat_level}")
    st.plotly_chart(threat_gauge(cumulative_reward), use_container_width=True)

# Lightweight polling-based refresh loop.
time.sleep(2)
st.rerun()
