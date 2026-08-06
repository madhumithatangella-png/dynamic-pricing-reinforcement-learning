"""About page explaining the Reinforcement Learning dynamic pricing formulation."""

import streamlit as st


def render(api_url: str):
    st.markdown("<h2 style='color: #F8FAFC;'>ℹ️ About the Intelligent Pricing Platform</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Learn about the underlying mathematical Reinforcement Learning formulation and demand simulator modeling.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<h3 style='color: #3B82F6;'>🤖 Reinforcement Learning Formulation</h3>", unsafe_allow_html=True)
        st.write(
            "The dynamic pricing problem is modeled as a finite-horizon **Markov Decision Process (MDP)**. "
            "An pricing agent makes daily pricing decisions to maximize cumulative expected revenue over a "
            "100-day sales booking window."
        )

        st.markdown(
            """
            *   **State Space ($S$)**: 5-dimensional discrete state:
                1.  `Remaining Inventory`: Rooms left to sell (0 to 50).
                2.  `Days Until Departure`: Days remaining in sales window (0 to 100).
                3.  `Arrival Season`: Categorized customer check-in season (Spring, Summer, Autumn, Winter).
                4.  `Booking Month`: Calendar month (1 to 12).
                5.  `Hotel Type`: City Hotel or Resort Hotel.
            *   **Action Space ($A$)**: 9 discrete price choices ranging from ₹3000 to ₹7000:
                `[3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]`
            *   **Reward Function ($R$)**: Combines direct booking revenue with unsold room penalties:
                $$Reward = Revenue - Penalty_{unsold}$$
            """
        )

    with col2:
        st.markdown("<h3 style='color: #8B5CF6;'>📈 Customer Demand Simulator</h3>", unsafe_allow_html=True)
        st.write(
            "Customer booking requests are generated dynamically based on historical check-in distributions "
            "and guest features. A machine learning pipeline estimates conversion probabilities:"
        )

        st.markdown(
            """
            1.  **Baseline Confirmation Probability**: A Logistic Regression model pre-trained on historical booking logs "
                predicts baseline success probability given guest demographics (lead time, market segment, agent, customer type).
            2.  **Price Elasticity Factor**: Room acceptance is modeled using a logistic pricing curve relative to "
                the expected market price (ADR) predicted via Ridge Regression:
                $$P(\\text{accept}) = \\frac{1}{1 + e^{\\alpha \\times \\frac{\\text{Price}_{\\text{offered}} - \\text{ADR}_{\\text{expected}}}{\\text{ADR}_{\\text{expected}}}}}$$
            3.  **Booking Conversion**: A booking is successfully recorded if:
                $$\\text{Random Roll} < P(\\text{baseline}) \\times P(\\text{accept})$$
            """
        )

    st.markdown("---")
    st.info(
        "**Technical Stack Summary:**\n\n"
        "- **Models**: Tabular Q-Table (NumPy) & Deep Q-Network (PyTorch Multi-Layer Perceptron)\n"
        "- **Optimizers**: Bellman Update & Adam Optimizer\n"
        "- **Inference**: High-speed REST calls mapped via FastAPI and served to Streamlit"
    )
