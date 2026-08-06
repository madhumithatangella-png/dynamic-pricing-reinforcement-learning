"""Training metrics page presenting interactive Plotly curves for DQN and Tabular Q-learning."""

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


def render(api_url: str):
    st.markdown("<h2 style='color: #F8FAFC;'>📈 Reinforcement Learning Training Center</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Evaluate agent policy optimization curves, network gradients loss decays, and Q-value distributions.</p>", unsafe_allow_html=True)
    st.markdown("---")

    REPORTS_DIR = Path("outputs/reports")
    q_metrics_path = REPORTS_DIR / "training_metrics.csv"
    dqn_metrics_path = REPORTS_DIR / "dqn_training_metrics.csv"

    # tabs
    tab_dqn, tab_q = st.tabs(["Deep Q-Network (DQN) Curves", "Tabular Q-Learning Curves"])

    with tab_dqn:
        if dqn_metrics_path.exists():
            df_dqn = pd.read_csv(dqn_metrics_path)
            
            # Key statistics summary
            st.markdown("<h4 style='color: #3B82F6;'>🚀 Training Status Summaries</h4>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            final_row = df_dqn.iloc[-1]
            col1.metric("Total Episodes", f"{len(df_dqn)}")
            col2.metric("Final Loss", f"{final_row['loss']:.4f}")
            col3.metric("Final Epsilon (Exploration)", f"{final_row['epsilon']:.4f}")
            col4.metric("Avg Q-Value Estimate", f"{final_row['average_q']:.2f}")
            
            st.markdown("---")
            
            # Chart 1: Reward Curve (Plotly)
            st.markdown("#### 🎯 Episodic Total Reward Convergence")
            df_dqn["rolling_reward"] = df_dqn["reward"].rolling(window=30, min_periods=1).mean()
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(x=df_dqn["episode"], y=df_dqn["reward"], name="Raw Reward", line=dict(color="rgba(59, 130, 246, 0.3)")))
            fig_reward.add_trace(go.Scatter(x=df_dqn["episode"], y=df_dqn["rolling_reward"], name="30-Ep Rolling Mean", line=dict(color="#3B82F6", width=2.5)))
            fig_reward.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC',
                margin=dict(l=20, r=20, t=30, b=20),
                height=350,
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_reward, use_container_width=True)

            # Columns for Loss and Histogram
            col_l, col_h = st.columns(2)
            
            with col_l:
                st.markdown("#### 📉 Neural Network MSE Loss Decay")
                df_loss = df_dqn[df_dqn["loss"] > 0]
                fig_loss = px.line(df_loss, x="episode", y="loss", title="Loss Curve (Log Scale)")
                fig_loss.update_traces(line=dict(color="#EF4444", width=1.5))
                fig_loss.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#F8FAFC',
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=300,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(type='log', gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            with col_h:
                st.markdown("#### 📊 Q-Value Expectation Distributions")
                fig_hist = px.histogram(df_dqn, x="average_q", nbins=30, title="Episode Average Q-Values")
                fig_hist.update_traces(marker=dict(color="#8B5CF6", line=dict(width=1, color="rgba(255,255,255,0.1)")))
                fig_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#F8FAFC',
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=300,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info(f"DQN training metrics file not found at `{dqn_metrics_path}`. Run `--phase 3` to train DQN models.")

    with tab_q:
        if q_metrics_path.exists():
            df_q = pd.read_csv(q_metrics_path)
            
            # Key statistics
            st.markdown("<h4 style='color: #8B5CF6;'>🚀 Q-Learning Status Summaries</h4>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            final_q = df_q.iloc[-1]
            col1.metric("Total Q-table Episodes", f"{len(df_q)}")
            col2.metric("Final Epsilon", f"{final_q['epsilon']:.4f}")
            col3.metric("Final Episode Reward", f"{final_q['reward']:.2f}")

            st.markdown("---")

            # Reward convergence (Plotly)
            st.markdown("#### 🎯 Episodic Total Reward Convergence")
            df_q["rolling_reward"] = df_q["reward"].rolling(window=30, min_periods=1).mean()
            fig_q_reward = go.Figure()
            fig_q_reward.add_trace(go.Scatter(x=df_q["episode"], y=df_q["reward"], name="Raw Reward", line=dict(color="rgba(139, 92, 246, 0.3)")))
            fig_q_reward.add_trace(go.Scatter(x=df_q["episode"], y=df_q["rolling_reward"], name="30-Ep Rolling Mean", line=dict(color="#8B5CF6", width=2.5)))
            fig_q_reward.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC',
                margin=dict(l=20, r=20, t=30, b=20),
                height=350,
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_q_reward, use_container_width=True)

            # Average Q-value chart
            st.markdown("#### 📊 Lookup Table Action Expected Values (Average Q)")
            fig_q_val = px.line(df_q, x="episode", y="average_q", title="Episodic Q-Value Mean Target")
            fig_q_val.update_traces(line=dict(color="#22C55E", width=1.5))
            fig_q_val.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC',
                margin=dict(l=20, r=20, t=40, b=20),
                height=300,
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_q_val, use_container_width=True)
        else:
            st.info(f"Q-learning metrics file not found at `{q_metrics_path}`. Run `--phase 2` to train Q-table models.")
