"""Prediction page providing interactive optimizations, XAI factor breakdowns, and confidence gauges."""

import streamlit as st
import requests
import plotly.graph_objects as go


def render(api_url: str):
    st.markdown("<h2 style='color: #F8FAFC;'>🎯 Room Price Optimization</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Real-time AI pricing recommendation, booking probability estimation, and demand justification factors.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_ctrl, col_res = st.columns([1, 1.2], gap="large")

    with col_ctrl:
        st.markdown("<h3 style='color: #3B82F6;'>⚙️ Input Parameters</h3>", unsafe_allow_html=True)
        
        # Wrapped inside a clean panel
        with st.container(border=True):
            hotel_type = st.selectbox(
                "Hotel Category",
                ["Resort Hotel", "City Hotel"],
                help="Category of the target hotel."
            )
            arrival_season = st.selectbox(
                "Arrival Season",
                ["Spring", "Summer", "Autumn", "Winter"],
                help="Target check-in season context."
            )
            booking_month = st.slider(
                "Check-in Month",
                min_value=1,
                max_value=12,
                value=6,
                help="Month corresponding to check-in."
            )
            remaining_inventory = st.slider(
                "Remaining Rooms Unsold",
                min_value=0,
                max_value=50,
                value=40,
                help="Rooms left unsold for the target check-in date. Total capacity is 50."
            )
            days_until_departure = st.slider(
                "Days Left (Booking Window)",
                min_value=0,
                max_value=100,
                value=30,
                help="Days left to sell the inventory. Max window is 100 days."
            )
            model_type = st.selectbox(
                "Inference Agent",
                ["DQN (Deep Q-Network)", "Q-Learning (Tabular)"],
                help="Select pricing agent policy to optimize room rates."
            )
            
            model_param = "dqn" if "DQN" in model_type else "q_learning"
            submit_btn = st.button("Generate Optimization Recommendation", type="primary", use_container_width=True)

    with col_res:
        st.markdown("<h3 style='color: #8B5CF6;'>📊 Optimized Recommendation</h3>", unsafe_allow_html=True)
        
        if submit_btn:
            payload = {
                "remaining_inventory": remaining_inventory,
                "days_until_departure": days_until_departure,
                "arrival_season": arrival_season,
                "booking_month": booking_month,
                "hotel_type": hotel_type,
                "model_type": model_param
            }
            
            with st.spinner("Invoking pricing agent models..."):
                try:
                    resp = requests.post(f"{api_url}/predict-price", json=payload)
                    if resp.status_code == 200:
                        result = resp.json()
                        
                        st.session_state["active_sim_data"] = {
                            "price": result["recommended_price"],
                            "hotel_type": hotel_type,
                            "arrival_season": arrival_season,
                            "booking_month": booking_month,
                            "days_until_departure": days_until_departure
                        }

                        # Display Recommended Rate
                        st.markdown(
                            f"""
                            <div class="glass-card" style="border: 2px solid #8B5CF6;">
                                <div class="metric-label" style="color: #8B5CF6;">RECOMMENDED ROOM RATE</div>
                                <div class="metric-value" style="font-size: 3rem;">₹{result['recommended_price']:.0f}</div>
                                <div style="display: flex; justify-content: space-between; margin-top: 15px;">
                                    <div>
                                        <div class="metric-label" style="font-size: 0.75rem;">Expected Revenue</div>
                                        <div style="font-size: 1.3rem; font-weight: 700; color: #22C55E;">₹{result['expected_revenue']:.2f}</div>
                                    </div>
                                    <div>
                                        <div class="metric-label" style="font-size: 0.75rem;">Current Occupancy</div>
                                        <div style="font-size: 1.3rem; font-weight: 700; color: #3B82F6;">{result['occupancy_estimate']*100:.0f}%</div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Booking Probability Gauge
                        prob = result["booking_probability"]
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#F8FAFC"},
                                'bar': {'color': "#3B82F6"},
                                'bgcolor': "rgba(30, 41, 59, 0.45)",
                                'borderwidth': 1,
                                'bordercolor': "rgba(255, 255, 255, 0.08)",
                                'steps': [
                                    {'range': [0, 40], 'color': '#EF4444'},
                                    {'range': [40, 75], 'color': '#F59E0B'},
                                    {'range': [75, 100], 'color': '#22C55E'}
                                ],
                            }
                        ))
                        fig.update_layout(
                            title={'text': "Booking Confirmation Probability (%)", 'font': {'color': '#F8FAFC', 'size': 14}},
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#F8FAFC',
                            margin=dict(l=30, r=30, t=50, b=10),
                            height=180
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Explainable AI Reason
                        st.markdown("<h4 style='color: #F8FAFC;'>💡 Explainable AI (XAI) Insight</h4>", unsafe_allow_html=True)
                        st.info(result["reason"])

                        # Top Factors List
                        st.markdown("#### Key Drivers Decision Weights")
                        col_f1, col_f2, col_f3 = st.columns(3)
                        with col_f1:
                            st.metric("Inventory Left", f"{remaining_inventory} Rooms", delta=result["inventory_status"], delta_color="inverse")
                        with col_f2:
                            st.metric("Time Urgency", f"{days_until_departure} Days Left", delta="Deadline" if days_until_departure < 15 else "Horizon")
                        with col_f3:
                            st.metric("Segment Demand", arrival_season, delta="Seasonal Peak" if arrival_season in ["Summer", "Winter"] else "Off-Peak")

                        # Direct Booking Simulator Action
                        st.markdown("---")
                        st.subheader("🎲 Simulate Customer Booking Transaction")
                        if st.button("Simulate Booking Response", type="primary", use_container_width=True):
                            sim_payload = st.session_state["active_sim_data"]
                            try:
                                sim_resp = requests.post(f"{api_url}/simulate-booking", json=sim_payload)
                                if sim_resp.status_code == 200:
                                    sim_res = sim_resp.json()
                                    if sim_res["booking_successful"]:
                                        st.balloons()
                                        st.success(f"🎉 Conversion Success! The customer booked the room at ₹{int(sim_payload['price'])}. (Probability was {sim_res['booking_probability']*100:.1f}%)")
                                    else:
                                        st.error(f"❌ Conversion Rejected. The customer did not confirm at ₹{int(sim_payload['price'])}. (Probability was {sim_res['booking_probability']*100:.1f}%)")
                            except Exception as e:
                                st.error(f"Failed to reach booking simulation backend: {e}")
                    else:
                        st.error(f"Failed to fetch optimal price: {resp.json().get('detail', 'Unknown API error')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend pricing service: {e}")
        else:
            st.info("Configure variables in the left panel and click 'Generate Optimization Recommendation' to run agent inference.")
