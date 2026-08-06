"""Evaluation page offering interactive policy testing, color-coded tables, and CSV exports."""

import streamlit as st
import requests
import pandas as pd
import time


def render(api_url: str):
    st.markdown("<h2 style='color: #F8FAFC;'>📋 Agent Policy Auditing</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Run isolated policy stress tests, analyze room sale efficiencies, and export reports.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Initialize evaluation history session state
    if "eval_history" not in st.session_state:
        st.session_state["eval_history"] = []

    col_opt, col_act = st.columns([1, 1], gap="medium")

    with col_opt:
        with st.container(border=True):
            st.markdown("<h4 style='color: #3B82F6;'>⚙️ Policy Selectors</h4>", unsafe_allow_html=True)
            model_type = st.selectbox(
                "Target Agent",
                ["DQN (Deep Q-Network)", "Q-Learning (Tabular)"],
                key="eval_model_select"
            )
            episodes = st.slider(
                "Evaluation Horizon Episodes",
                min_value=5,
                max_value=100,
                value=20,
                help="Number of full 100-day simulation cycles to average policy performance."
            )
            
            model_param = "dqn" if "DQN" in model_type else "q_learning"
            run_btn = st.button("Execute Policy Evaluation Run", type="primary", use_container_width=True)

    with col_act:
        with st.container(border=True):
            st.markdown("<h4 style='color: #8B5CF6;'>🛠️ Table Management</h4>", unsafe_allow_html=True)
            st.write("Maintain audit runs log across different parameters and export reports directly.")
            clear_btn = st.button("Clear Evaluation Logs History", use_container_width=True)
            if clear_btn:
                st.session_state["eval_history"] = []
                st.success("Evaluation history logs cleared successfully.")

    if run_btn:
        with st.spinner(f"Evaluating {model_type}..."):
            try:
                resp = requests.post(
                    f"{api_url}/evaluate-policy",
                    json={"model_type": model_param, "episodes": episodes}
                )
                if resp.status_code == 200:
                    result = resp.json()
                    
                    # Append result to session state history
                    st.session_state["eval_history"].append({
                        "Run ID": f"RUN_{int(time.time()) % 100000}",
                        "Timestamp": time.strftime("%H:%M:%S"),
                        "Model": model_param.upper(),
                        "Episodes Tested": result["episodes_evaluated"],
                        "Avg Reward": round(result["average_reward"], 2),
                        "Rooms Sold": result["rooms_sold"],
                        "Avg Occupancy Rate": round(result["average_occupancy"], 4),
                        "Avg Revenue (INR)": round(result["average_revenue"], 2),
                    })
                    st.success(f"Audit completed. Added {model_param.upper()} evaluation parameters to database.")
                else:
                    st.error(f"Evaluation failed: {resp.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed to connect to API evaluation endpoint: {e}")

    # Display History Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #F8FAFC;'>📋 Policy Audit Records</h3>", unsafe_allow_html=True)
    
    if st.session_state["eval_history"]:
        df = pd.DataFrame(st.session_state["eval_history"])
        
        # Color coding formatting using Pandas styler
        # Color code: Higher revenues are shaded green, lower revenues shaded orange/red
        df_styled = df.style.background_gradient(
            cmap="Greens", subset=["Avg Revenue (INR)"]
        ).background_gradient(
            cmap="Purples", subset=["Avg Occupancy Rate"]
        ).format({
            "Avg Revenue (INR)": "₹{:,.2f}",
            "Avg Occupancy Rate": "{:.2%}"
        })

        st.dataframe(df_styled, use_container_width=True)

        # Download CSV Button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Audit Logs as CSV Report",
            data=csv_data,
            file_name=f"hotel_pricing_policy_audit_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No evaluations run yet. Configure the selectors above and execute an audit run to populate the table.")
