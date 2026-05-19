"""
Smartphone Price Prediction & Market Intelligence Dashboard.

A modern, visually striking Streamlit web application.
Operates strictly as a frontend client via REST requests mapping 
to the decoupled FastAPI backend ('backend_api.py').
"""
import streamlit as st
import pandas as pd
import requests
import os
import plotly.express as px

from components.style import inject_custom_css
from components.sidebar import render_sidebar
from components.metrics import render_metric_card
from components.charts import apply_plotly_theme
from components.chat import render_chat_interface

# ---------------------------------------------------------
# Page Configuration & Modern UI Injection
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smartphone Market AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@st.cache_data(ttl=3600)
def load_market_data() -> pd.DataFrame:
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/market-data", timeout=15)
        resp.raise_for_status()
        records = resp.json().get("data", [])
        if not records:
            return pd.DataFrame(columns=["Brand", "Price", "ram_gb", "battery_mah", "camera_mp", "Processor", "Smartphone_Name"])
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame(columns=["Brand", "Price", "ram_gb", "battery_mah", "camera_mp", "Processor", "Smartphone_Name"])

df_visuals = load_market_data()

# Global Navigation
selected_page = render_sidebar()

# ── MAIN VIEW ROUTING ─────────────────────────────────────────────

if selected_page == "📈 Market Analytics":
    st.markdown("## Market Analytics")
    st.markdown("<p style='color:#8E8E93;'>Decoupled ML Predictions & RAG Analytics</p>", unsafe_allow_html=True)
    
    if len(df_visuals) > 0:
        # High-Level KPIs
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Total Devices Tracked", f"{len(df_visuals):,}")
        with col2:
            avg_price = df_visuals["Price"].mean()
            render_metric_card("Average Market Price", f"${avg_price:,.2f}")
        with col3:
            top_brand = df_visuals["Brand"].mode()[0] if not df_visuals.empty else "N/A"
            render_metric_card("Most Dominant Brand", top_brand)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs for Layout
        tab_overview, tab_compare = st.tabs(["Ecosystem Overview", "Hardware Analysis"])
        
        with tab_overview:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            df_visuals["Operating System"] = df_visuals["Brand"].apply(lambda x: "iOS" if x == "Apple" else "Android")
            fig_sunburst = px.sunburst(
                df_visuals,
                path=["Operating System", "Brand"],
                values="Price",
                color="Price",
                color_continuous_scale=px.colors.sequential.Plasma,
                title="Global Revenue Distribution"
            )
            fig_sunburst = apply_plotly_theme(fig_sunburst)
            fig_sunburst.update_traces(hovertemplate='<b>%{label}</b><br>Value: $%{value:,.0f}')
            st.plotly_chart(fig_sunburst, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab_compare:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            radar_brands = st.multiselect(
                "Select two brands to compare:",
                options=df_visuals["Brand"].unique(),
                default=["Apple", "Samsung"],
                max_selections=2
            )
            
            if len(radar_brands) == 2:
                radar_df = df_visuals[df_visuals["Brand"].isin(radar_brands)]
                radar_agg = radar_df.groupby("Brand")[["ram_gb", "battery_mah", "Price"]].mean().reset_index()
                
                for col in ["ram_gb", "battery_mah", "Price"]:
                    max_val = df_visuals[col].max()
                    radar_agg[col + "_norm"] = radar_agg[col] / max_val
                
                radar_melt = radar_agg.melt(id_vars=["Brand"], value_vars=["ram_gb_norm", "battery_mah_norm", "Price_norm"], var_name="Metric", value_name="Score")
                radar_melt["Metric"] = radar_melt["Metric"].str.replace("_norm", "").str.upper()
                
                fig_radar = px.line_polar(
                    radar_melt, 
                    r="Score", 
                    theta="Metric", 
                    color="Brand", 
                    line_close=True,
                    title="Hardware vs Valuation (Normalized)",
                    color_discrete_sequence=["#2563EB", "#7C3AED"]
                )
                fig_radar = apply_plotly_theme(fig_radar)
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)))
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Select exactly 2 brands above to render the Radar Analysis.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Production dataset missing. Please sync live data.")

elif selected_page == "⚖️ Smartphone Comparison":
    st.markdown("## Smartphone Comparison Model")
    st.markdown("<p style='color:#8E8E93;'>Hardware Valuation Engine natively via Random Forest</p>", unsafe_allow_html=True)
    
    valid_brands = df_visuals["Brand"].unique().tolist() if len(df_visuals) > 0 else ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Vivo", "Oppo", "Motorola"]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_brand, col_proc = st.columns(2)
    with col_brand:
        input_brand_dynamic = st.selectbox("Manufacturer Brand", valid_brands, key="dynamic_brand")
    with col_proc:
        input_processor_dynamic = st.selectbox("Processor", ["A15 Bionic", "A16 Bionic", "A17 Pro", "Snapdragon 8 Gen 2", "Snapdragon 8 Gen 3", "Exynos 2400", "MediaTek Dimensity 9000", "Unknown"])
        
    if len(df_visuals) > 0:
        brand_df = df_visuals[df_visuals["Brand"] == input_brand_dynamic]
        if len(brand_df) > 0:
            ram_min, ram_max = float(brand_df["ram_gb"].min()), float(brand_df["ram_gb"].max())
            batt_min, batt_max = float(brand_df["battery_mah"].min()), float(brand_df["battery_mah"].max())
        else:
            ram_min, ram_max = 2.0, 24.0
            batt_min, batt_max = 2000.0, 7000.0
    else:
        ram_min, ram_max = 2.0, 24.0
        batt_min, batt_max = 2000.0, 7000.0
        
    if ram_min == ram_max: ram_max += 2.0
    if batt_min == batt_max: batt_max += 500.0

    with st.form("prediction_form"):
        col_form1, col_form2, col_form3 = st.columns(3)
        with col_form1:
            input_ram = st.slider("RAM (GB)", float(ram_min), float(ram_max), float(ram_min) + ((ram_max-ram_min)/2), step=1.0)
        with col_form2:
            input_battery = st.slider("Battery (mAh)", float(batt_min), float(batt_max), float(batt_min)+((batt_max-batt_min)/2), step=50.0)
        with col_form3:
            input_camera = st.slider("Camera (MP)", 8.0, 200.0, 50.0, step=2.0)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submit_prediction = st.form_submit_button("🌩️ Calculate Algorithmic Valuation")
        
    st.markdown('</div>', unsafe_allow_html=True)
        
    if submit_prediction:
        payload = {
            "Brand": input_brand_dynamic,
            "Processor": input_processor_dynamic,
            "ram_gb": input_ram,
            "battery_mah": input_battery,
            "camera_mp": input_camera
        }
        with st.spinner("Executing Model Inferences..."):
            try:
                response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    predicted_price = data.get("predicted_price", 0.0)
                    
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center; background: linear-gradient(135deg, rgba(37,99,235,0.05) 0%, rgba(124,58,237,0.05) 100%); border-color: rgba(37,99,235,0.3);">
                        <h3 style="color: #8E8E93; margin: 0; font-size: 1.2rem; font-weight: 500;">Estimated Trading Value</h3>
                        <h1 style="margin: 10px 0 0 0; font-size: 4.5rem; font-weight: 800; background: linear-gradient(135deg, var(--primary-color, #2563EB) 0%, #7C3AED 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">${predicted_price:,.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Prediction Pipeline Error: {e}")

elif selected_page == "🤖 AI Assistant":
    render_chat_interface(API_BASE_URL)

elif selected_page == "⚙️ System Settings":
    st.markdown("## System Settings")
    st.markdown("<p style='color:#8E8E93;'>Backend Service Health and Connectivity</p>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    try:
        requests.get(f"{API_BASE_URL}/docs", timeout=3)
        st.success("📡 Backend: Online")
    except Exception:
        st.error("📡 Backend: Offline")
        
    st.info(f"🗄️ {len(df_visuals):,} devices tracked")
    st.markdown('</div>', unsafe_allow_html=True)

elif selected_page == "🔄 Sync Live Data":
    st.markdown("## Sync Live Market Data")
    st.markdown("<p style='color:#8E8E93;'>Manage your data pipeline and fetch live updates</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_sync1, col_sync2 = st.columns([2, 1])
    with col_sync1:
        st.markdown("### 🔄 Sync Live Market Data")
        st.markdown("Fetch the latest device information from DuckDuckGo/ChromaDB and update the analytics dashboard natively.")
    with col_sync2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start Live Sync", use_container_width=True, type="primary"):
            try:
                trigger = requests.post(f"{API_BASE_URL}/api/v1/trigger-pipeline", timeout=10).json()
                if trigger.get("status") == "already_running":
                    st.warning("⏳ Sync already running.")
                else:
                    with st.spinner("Fetching data & retraining..."):
                        import time
                        success = False
                        result_msg = "Timeout."
                        for _ in range(60):
                            time.sleep(2)
                            try:
                                poll = requests.get(f"{API_BASE_URL}/api/v1/pipeline-status", timeout=5).json()
                            except requests.exceptions.RequestException:
                                continue
                            if not poll.get("running", True):
                                result_msg = poll.get("last_result", "Done.")
                                success = "failed" not in result_msg.lower()
                                break
                    if success:
                        st.toast("✅ Sync complete!", icon="✅")
                        load_market_data.clear()
                        st.rerun()
                    else:
                        st.error(f"❌ Sync failed: {result_msg}")
            except Exception as e:
                st.error("❌ Sync Error.")
    st.markdown('</div>', unsafe_allow_html=True)
