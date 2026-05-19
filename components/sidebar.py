import streamlit as st
import requests

def render_sidebar(df_length, api_base_url, load_data_func=None):
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; font-weight: 800; letter-spacing: -0.05em;'>📱 Market AI</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 🧭 Navigation")
        if st.button("📊 Global Dashboard", use_container_width=True):
            st.session_state.current_view = 'dashboard'
            st.rerun()
        if st.button("📈 Market Analytics", use_container_width=True):
            st.session_state.current_view = 'analytics'
            st.rerun()
        if st.button("🌩️ Price Predictor", use_container_width=True):
            st.session_state.current_view = 'price_predictor'
            st.rerun()
        if st.button("🤖 AI Assistant", use_container_width=True):
            st.session_state.current_view = 'ai'
            st.rerun()
        if st.button("🔄 Data Sync & Settings", use_container_width=True):
            st.session_state.current_view = 'sync'
            st.rerun()
        
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("#### System Status")
        
        try:
            requests.get(f"{api_base_url}/docs", timeout=3)
            st.success("📡 Backend: Online")
        except Exception:
            st.error("📡 Backend: Offline")
            
        st.info(f"🗄️ {df_length:,} devices tracked")
