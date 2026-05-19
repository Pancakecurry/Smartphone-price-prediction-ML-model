import streamlit as st

def render_metric_card(label, value):
    st.markdown(f"""
    <div class="glass-card" style="text-align: center; padding: 20px;">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)
