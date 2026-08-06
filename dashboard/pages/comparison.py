"""Model comparison page with interactive Plotly benchmarks for all pricing agents."""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px


def render(api_url: str):
    st.markdown("<h2 style='color: #F8FAFC;'>⚖️ Multi-Agent Benchmark Comparisons</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Benchmark Deep Q-Networks (DQN), Tabular Q-learning, and heuristic pricing models side-by-side.</p>", unsafe_allow_html=True)
    st.markdown("---")

    episodes = st.slider(
        "Simulation Episodes per Agent",
        min_value=5,
        max_value=50,
        value=20,
        help="Higher values give more statistically stable results but run longer in the simulator."
    )

    if st.button("Run Multi-Agent Simulation Benchmark", type="primary", use_container_width=True):
        with st.spinner("Executing agent trajectories in environments..."):
            try:
                resp = requests.post(f"{api_url}/compare-models", json={"episodes": episodes})
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    
                    if results:
                        df = pd.DataFrame(results)
                        
                        # Best performers
                        best_rev = df.loc[df["Total Revenue"].idxmax()]
                        best_occ = df.loc[df["Occupancy Rate"].idxmax()]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(
                                f"""
                                <div class="glass-card" style="border-left: 4px solid #22C55E;">
                                    <div class="metric-label">Revenue Leader</div>
                                    <div class="metric-value">{best_rev['Agent']}</div>
                                    <div class="metric-trend trend-up">₹{best_rev['Total Revenue']:,.0f} Total Revenue</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        with col2:
                            st.markdown(
                                f"""
                                <div class="glass-card" style="border-left: 4px solid #3B82F6;">
                                    <div class="metric-label">Occupancy Leader</div>
                                    <div class="metric-value">{best_occ['Agent']}</div>
                                    <div class="metric-trend trend-up" style="color: #3B82F6;">{best_occ['Occupancy Rate']*100:.1f}% Avg Occupancy</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.markdown("<br>", unsafe_allow_html=True)

                        # Row 1 Charts: Revenue and Occupancy
                        col_r, col_o = st.columns(2)
                        
                        with col_r:
                            fig_rev = px.bar(
                                df, 
                                x="Agent", 
                                y="Total Revenue", 
                                color="Agent",
                                title="Cumulative Revenue (INR)",
                                color_discrete_sequence=px.colors.qualitative.Plotly
                            )
                            fig_rev.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='#F8FAFC',
                                margin=dict(l=20, r=20, t=40, b=20),
                                showlegend=False,
                                xaxis=dict(showgrid=False),
                                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                            )
                            st.plotly_chart(fig_rev, use_container_width=True)

                        with col_o:
                            fig_occ = px.bar(
                                df, 
                                x="Agent", 
                                y="Occupancy Rate", 
                                color="Agent",
                                title="Average Occupancy Rate",
                                color_discrete_sequence=px.colors.qualitative.Vivid
                            )
                            fig_occ.yaxis.set_major_formatter(px.ticks.PercentFormat())
                            fig_occ.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='#F8FAFC',
                                margin=dict(l=20, r=20, t=40, b=20),
                                showlegend=False,
                                xaxis=dict(showgrid=False),
                                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                            )
                            st.plotly_chart(fig_occ, use_container_width=True)

                        # Row 2 Charts: Acceptance Rate and Inference Speed
                        col_a, col_i = st.columns(2)
                        
                        with col_a:
                            fig_acc = px.bar(
                                df, 
                                x="Agent", 
                                y="Booking Acceptance Rate", 
                                color="Agent",
                                title="Booking Acceptance Rate",
                                color_discrete_sequence=px.colors.qualitative.Dark24
                            )
                            fig_acc.yaxis.set_major_formatter(px.ticks.PercentFormat())
                            fig_acc.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='#F8FAFC',
                                margin=dict(l=20, r=20, t=40, b=20),
                                showlegend=False,
                                xaxis=dict(showgrid=False),
                                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                            )
                            st.plotly_chart(fig_acc, use_container_width=True)

                        with col_i:
                            fig_inf = px.bar(
                                df, 
                                x="Agent", 
                                y="Inference Time", 
                                color="Agent",
                                title="Inference Latency (Milliseconds)",
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            fig_inf.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='#F8FAFC',
                                margin=dict(l=20, r=20, t=40, b=20),
                                showlegend=False,
                                xaxis=dict(showgrid=False),
                                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                            )
                            st.plotly_chart(fig_inf, use_container_width=True)

                        st.markdown("---")
                        st.subheader("Performance Metric Breakdown Table")
                        
                        # Styled dataframe display
                        df_table = df.copy()
                        df_styled = df_table.style.format({
                            "Total Revenue": "₹{:,.2f}",
                            "Revenue per Episode": "₹{:,.2f}",
                            "Average Reward": "{:.2f}",
                            "Booking Acceptance Rate": "{:.1%}",
                            "Occupancy Rate": "{:.1%}",
                            "Average Room Price": "₹{:,.2f}",
                            "Inference Time": "{:.2f} ms"
                        })
                        st.dataframe(df_styled, use_container_width=True)

                    else:
                        st.warning("No metrics returned by benchmarking services.")
                else:
                    st.error(f"Benchmarking request failed: {resp.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed to connect to API benchmarking service: {e}")
    else:
        st.info("Select simulated episodes and click 'Run Multi-Agent Simulation Benchmark' to trigger concurrently.")
