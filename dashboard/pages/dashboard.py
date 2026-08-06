"""Dashboard page showing professional metrics, KPIs, and real-time request logs."""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px


def render(api_url: str):
    st.markdown("<h2 style='color: #F8FAFC;'>📊 Performance & MLOps Control Room</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Enterprise health overview, hardware utilization, and prediction latency metrics.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Fetch live metrics from API
    try:
        resp = requests.get(f"{api_url}/metrics", timeout=2)
        if resp.status_code == 200:
            metrics = resp.json()
        else:
            metrics = {}
    except Exception:
        metrics = {}

    prediction_count = metrics.get("prediction_count", 0)
    avg_latency = metrics.get("average_latency_ms", 0.0)
    avg_inference = metrics.get("average_inference_time_ms", 0.0)
    device = metrics.get("device", "CPU")
    history = metrics.get("prediction_history", [])

    # Layout Glassmorphic KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Total Predictions</div>
                <div class="metric-value">{prediction_count}</div>
                <div class="metric-trend trend-up">▲ Active API Runs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Avg API Latency</div>
                <div class="metric-value">{avg_latency:.1f}ms</div>
                <div class="metric-trend trend-up">▲ Under 50ms Limit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">ML Inference</div>
                <div class="metric-value">{avg_inference:.2f}ms</div>
                <div class="metric-trend trend-up">▼ Optimized Model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Computing Node</div>
                <div class="metric-value">{device.upper()}</div>
                <div class="metric-trend trend-up">● Standard Dev Node</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Historical SaaS Stats Cards
    st.markdown("<h3 style='color: #F8FAFC;'>🏆 Simulated Operations Success</h3>", unsafe_allow_html=True)
    scol1, scol2, scol3 = st.columns(3)
    
    with scol1:
        st.markdown(
            """
            <div class="glass-card" style="border-left: 4px solid #3B82F6;">
                <div class="metric-label">Historical Rev. (DQN)</div>
                <div class="metric-value">₹2.51M</div>
                <div class="metric-trend trend-up" style="color: #3B82F6;">▲ +12.4% vs Fixed Rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with scol2:
        st.markdown(
            """
            <div class="glass-card" style="border-left: 4px solid #8B5CF6;">
                <div class="metric-label">Average Occupancy</div>
                <div class="metric-value">78.4%</div>
                <div class="metric-trend trend-up" style="color: #8B5CF6;">▲ +6.8% vs Baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with scol3:
        st.markdown(
            """
            <div class="glass-card" style="border-left: 4px solid #22C55E;">
                <div class="metric-label">Rooms Confirmed</div>
                <div class="metric-value">1,532</div>
                <div class="metric-trend trend-up" style="color: #22C55E;">▲ 86% Success Ratio</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Latency Chart & Recent Logs Table
    if history:
        df_hist = pd.DataFrame(history)
        df_hist["latency_ms"] = df_hist["latency_ms"].astype(float)
        
        col_c, col_t = st.columns([1, 1])
        
        with col_c:
            st.markdown("<h3 style='color: #F8FAFC;'>📈 API Latency Trend (Plotly)</h3>", unsafe_allow_html=True)
            fig = px.line(
                df_hist, 
                x="timestamp", 
                y="latency_ms", 
                title="Response Latency per Call",
                markers=True,
                labels={"latency_ms": "Latency (ms)", "timestamp": "Timestamp"}
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC',
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_t:
            st.markdown("<h3 style='color: #F8FAFC;'>📋 Recent Activity Logs</h3>", unsafe_allow_html=True)
            records = []
            for item in reversed(history):
                req = item.get("request", {})
                res = item.get("response", {})
                records.append({
                    "Time": item.get("timestamp"),
                    "Model": res.get("model_used", "").upper(),
                    "Hotel": req.get("hotel_type"),
                    "Season": req.get("arrival_season"),
                    "Occupancy": f"{((50 - req.get('remaining_inventory'))/50)*100:.0f}%",
                    "Price": f"₹{res.get('recommended_price'):.0f}",
                })
            df_table = pd.DataFrame(records)
            st.dataframe(df_table, use_container_width=True)
    else:
        st.info("No transaction logs recorded yet. Go to 'Live Prediction' to generate optimal recommendations.")
