import streamlit as st
import requests
import time

def render_sidebar(df_length, api_base_url, load_data_func):
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; font-weight: 800; letter-spacing: -0.05em;'>📱 Market AI</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["Market Analytics", "Price Predictor", "AI Assistant"],
            label_visibility="hidden"
        )
        
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("#### System Status")
        
        try:
            requests.get(f"{api_base_url}/docs", timeout=3)
            st.success("📡 Backend: Online")
        except Exception:
            st.error("📡 Backend: Offline")
            
        st.info(f"🗄️ {df_length:,} devices tracked")
        
        if st.button("🔄 Sync Live Data", use_container_width=True):
            try:
                trigger = requests.post(f"{api_base_url}/api/v1/trigger-pipeline", timeout=10).json()
                if trigger.get("status") == "already_running":
                    st.warning("⏳ Sync already running.")
                else:
                    with st.spinner("Fetching data & retraining..."):
                        success = False
                        result_msg = "Timeout."
                        for _ in range(60):
                            time.sleep(2)
                            try:
                                poll = requests.get(f"{api_base_url}/api/v1/pipeline-status", timeout=5).json()
                            except requests.exceptions.RequestException:
                                continue
                            if not poll.get("running", True):
                                result_msg = poll.get("last_result", "Done.")
                                success = "failed" not in result_msg.lower()
                                break
                    if success:
                        st.toast("✅ Sync complete!", icon="✅")
                        load_data_func.clear()
                        st.rerun()
                    else:
                        st.error(f"❌ Sync failed: {result_msg}")
            except Exception as e:
                st.error("❌ Sync Error.")
        return page
