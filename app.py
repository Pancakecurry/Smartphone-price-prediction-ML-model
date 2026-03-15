"""
Smartphone Price Prediction & Market Intelligence Dashboard.

A modern, visually striking Streamlit web application.
Operates strictly as a frontend client via REST requests mapping 
to the decoupled FastAPI backend ('backend_api.py').
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time

# ---------------------------------------------------------
# Page Configuration & Modern UI Injection
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smartphone Market AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Sleek CSS
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Default Menus */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    .main-title {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        padding-bottom: 20px;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #A0AEC0;
        margin-bottom: 40px;
        font-size: 1.2rem;
    }
    
    /* Metric Cards Glassmorphism */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #92FE9D;
    }
    .metric-card-custom {
        background: rgba(30, 37, 48, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 30px;
        transition: transform 0.3s ease;
    }
    .metric-card-custom:hover {
        transform: translateY(-5px);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #0E1117;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 201, 255, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 8px 8px 0px 0px;
        gap: 10px;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E2530 !important;
        border-bottom: 3px solid #00C9FF !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# API Target Configuration
# ---------------------------------------------------------
API_BASE_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------
# Data Loading — fully via REST API, no local file access
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_market_data() -> pd.DataFrame:
    """
    Fetches the full market dataset from the FastAPI backend.
    Cached for 1 hour (ttl=3600). Call st.cache_data.clear() to invalidate.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/market-data", timeout=15)
        resp.raise_for_status()
        records = resp.json().get("data", [])
        if not records:
            return pd.DataFrame(columns=["Brand", "Price", "ram_gb", "battery_mah", "camera_mp", "Processor", "Smartphone_Name"])
        return pd.DataFrame(records)
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Backend offline — start `uvicorn backend_api:app --reload` to load live data.", icon="🔌")
        return pd.DataFrame(columns=["Brand", "Price", "ram_gb", "battery_mah", "camera_mp", "Processor", "Smartphone_Name"])
    except Exception as e:
        st.error(f"Failed to load market data: {e}")
        return pd.DataFrame(columns=["Brand", "Price", "ram_gb", "battery_mah", "camera_mp", "Processor", "Smartphone_Name"])


df_visuals = load_market_data()



# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown('<h1 class="main-title">Smartphone Market AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Decoupled ML Predictions & RAG Analytics powered by Llama 3</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# System & Data Controls  (always visible on main canvas)
# ---------------------------------------------------------
with st.expander("⚙️ System & Data Controls", expanded=True):
    ctrl_col, status_col, info_col = st.columns([2, 3, 2])

    with ctrl_col:
        if st.button("🔄 Sync Live Market Data", use_container_width=True):
            try:
                # Step 1: Fire the pipeline trigger (returns instantly)
                trigger = requests.post(
                    f"{API_BASE_URL}/api/v1/trigger-pipeline", timeout=10
                ).json()

                if trigger.get("status") == "already_running":
                    st.warning("⏳ A pipeline sync is already running. Check back shortly.")
                else:
                    # Step 2: Block the UI with a spinner while we poll for completion
                    with st.spinner(
                        "🔬 Fetching live market data, sanitizing inputs, and retraining ML models… "
                        "Please wait (up to 2 min)."
                    ):
                        success = False
                        result_msg = "Pipeline timed out before completion."
                        for _ in range(60):      # poll every 2s for up to 120s
                            time.sleep(2)
                            try:
                                poll = requests.get(
                                    f"{API_BASE_URL}/api/v1/pipeline-status", timeout=5
                                ).json()
                            except requests.exceptions.RequestException:
                                continue        # backend hiccup — keep polling

                            if not poll.get("running", True):
                                result_msg = poll.get("last_result", "Done.")
                                success = "failed" not in result_msg.lower()
                                break

                    # Step 3: Only clear cache and rerun on confirmed success
                    if success:
                        st.toast("✅ Sync complete! Reloading fresh data…", icon="✅")
                        load_market_data.clear()   # invalidate TTL cache
                        st.rerun()
                    else:
                        # Cache intentionally preserved so historical data remains visible
                        st.error(
                            f"❌ Pipeline sync failed — historical data retained.\n\n"
                            f"**Reason:** {result_msg}"
                        )
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend offline. Start `uvicorn backend_api:app --reload`.")
            except requests.exceptions.Timeout:
                st.error("❌ Trigger request timed out. Is the backend responding?")


    with status_col:
        try:
            requests.get(f"{API_BASE_URL}/docs", timeout=3)
            st.success("📡 Backend: **Online** ✅")
        except Exception:
            st.error("📡 Backend: **Offline** ❌ — run `uvicorn backend_api:app --reload`")

    with info_col:
        st.info(f"🗄️ **{len(df_visuals):,}** devices in dataset")

# Define Tabs
tab_market, tab_predict, tab_chat = st.tabs([
    "📊 Market Intelligence", 
    "🔮 Price Predictor", 
    "🤖 Groq AI Analyst"
])

# =========================================================
# TAB 1: Market Intelligence (Plotly Analytics)
# =========================================================
with tab_market:
    st.markdown("### Ecosystem Overview")

    if len(df_visuals) > 0:
        # High-Level KPIs
        col1, col2, col3 = st.columns(3)
        pd_df = df_visuals  # already a Pandas DataFrame
        
        with col1:
            st.markdown('<div class="metric-card-custom">', unsafe_allow_html=True)
            st.metric("Total Devices Tracked", f"{len(pd_df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card-custom">', unsafe_allow_html=True)
            avg_price = pd_df["Price"].mean()
            st.metric("Average Market Price", f"${avg_price:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card-custom">', unsafe_allow_html=True)
            top_brand = pd_df["Brand"].mode()[0] if not pd_df.empty else "N/A"
            st.metric("Most Dominant Brand", top_brand)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.divider()
        
        # Interactive Plotly Charts
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### Market Ecosystem Hierarchy")
            # Create a fallback "Operating System" column for Sunburst hierarchy since we dropped it earlier
            pd_df["Operating System"] = pd_df["Brand"].apply(lambda x: "iOS" if x == "Apple" else "Android")
            
            # Sunburst Chart: OS -> Brand -> Average Price
            fig_sunburst = px.sunburst(
                pd_df,
                path=["Operating System", "Brand"],
                values="Price",
                color="Price",
                color_continuous_scale=px.colors.sequential.Plasma,
                template="plotly_dark",
                title="Global Revenue Distribution"
            )
            fig_sunburst.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)", 
                font_color="#FAFAFA"
            )
            # Format hover data to currency
            fig_sunburst.update_traces(hovertemplate='<b>%{label}</b><br>Value: $%{value:,.0f}')
            st.plotly_chart(fig_sunburst, width="stretch")
            
        with chart_col2:
            st.markdown("#### Competitor Hardware Analysis")
            
            # Radar Chart UI Selection
            radar_brands = st.multiselect(
                "Select two brands to compare:",
                options=pd_df["Brand"].unique(),
                default=["Apple", "Samsung"],
                max_selections=2
            )
            
            if len(radar_brands) == 2:
                # Aggregate averages for Radar
                radar_df = pd_df[pd_df["Brand"].isin(radar_brands)]
                radar_agg = radar_df.groupby("Brand")[["ram_gb", "battery_mah", "Price"]].mean().reset_index()
                
                # Normalize values for Radar visual symmetry (0-1 scale visually)
                for col in ["ram_gb", "battery_mah", "Price"]:
                    max_val = pd_df[col].max()
                    radar_agg[col + "_norm"] = radar_agg[col] / max_val
                
                # Melt for Plotly Polar
                radar_melt = radar_agg.melt(id_vars=["Brand"], value_vars=["ram_gb_norm", "battery_mah_norm", "Price_norm"], var_name="Metric", value_name="Score")
                radar_melt["Metric"] = radar_melt["Metric"].str.replace("_norm", "").str.upper()
                
                fig_radar = px.line_polar(
                    radar_melt, 
                    r="Score", 
                    theta="Metric", 
                    color="Brand", 
                    line_close=True,
                    template="plotly_dark",
                    title="Hardware vs Valuation (Normalized)",
                    color_discrete_sequence=["#00C9FF", "#92FE9D"]
                )
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=False)),
                    plot_bgcolor="rgba(0,0,0,0)", 
                    paper_bgcolor="rgba(0,0,0,0)", 
                    font_color="#FAFAFA"
                )
                st.plotly_chart(fig_radar, width="stretch")
            else:
                st.info("Select exactly 2 brands above to render the Radar Analysis.")
    else:
        st.warning("Production Parquet dataset missing. Complete Phase 1 & 2 to view dynamic analytics.")

# =========================================================
# TAB 2: Price Predictor (FastAPI ML Integration)
# =========================================================
with tab_predict:
    st.markdown("### Hardware Valuation Engine")
    st.markdown("Configure theoretical specifications below to infer global market value natively via Random Forest.")
    # Determine Dynamic UI Bounds based on Brand native data
    valid_brands = df_visuals["Brand"].unique().tolist() if len(df_visuals) > 0 else ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Vivo", "Oppo", "Motorola"]

    # Extract dynamic selectors outside the form for immediate reactivity
    st.markdown("##### 1. Select Manufacturer")
    input_brand_dynamic = st.selectbox("Manufacturer Brand", valid_brands, key="dynamic_brand")
    input_processor_dynamic = st.selectbox("Processor (Optional context)", ["A15 Bionic", "A16 Bionic", "A17 Pro", "Snapdragon 8 Gen 2", "Snapdragon 8 Gen 3", "Exynos 2400", "MediaTek Dimensity 9000", "Unknown"])
    
    # Calculate Data-Driven Bounds for Sliders gracefully
    if len(df_visuals) > 0:
        brand_df = df_visuals[df_visuals["Brand"] == input_brand_dynamic]
        if len(brand_df) > 0:
            ram_min = float(brand_df["ram_gb"].min())
            ram_max = float(brand_df["ram_gb"].max())
            batt_min = float(brand_df["battery_mah"].min())
            batt_max = float(brand_df["battery_mah"].max())
        else:
            ram_min, ram_max = 2.0, 24.0
            batt_min, batt_max = 2000.0, 7000.0
    else:
        ram_min, ram_max = 2.0, 24.0
        batt_min, batt_max = 2000.0, 7000.0
        
    # Prevent slider crash if min == max
    if ram_min == ram_max: ram_max += 2.0
    if batt_min == batt_max: batt_max += 500.0

    st.markdown("##### 2. Configure Hardware Parameters")
    with st.form("prediction_form"):
        col_form3, col_form4 = st.columns(2)
        
        with col_form3:
            input_ram = st.slider("RAM Capacity (GB)", min_value=float(ram_min), max_value=float(ram_max), value=float(ram_min) + ((ram_max-ram_min)/2), step=1.0)
            
        with col_form4:
            input_battery = st.slider("Battery Capacity (mAh)", min_value=float(batt_min), max_value=float(batt_max), value=float(batt_min)+((batt_max-batt_min)/2), step=50.0)
            
        input_camera = st.slider("Primary Camera Matrix (MP)", min_value=8.0, max_value=200.0, value=50.0, step=2.0)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submit_prediction = st.form_submit_button("🌩️ Calculate Algorithmic Valuation")
        
    if submit_prediction:
        # Build Payload matching FastApi `PhoneSpecs` Pydantic model
        payload = {
            "Brand": input_brand_dynamic,
            "Processor": input_processor_dynamic,
            "ram_gb": input_ram,
            "battery_mah": input_battery,
            "camera_mp": input_camera
        }
        
        with st.spinner("Executing Scikit-Learn Matrix inversions..."):
            try:
                # Issue HTTP POST dynamically decoupled
                response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    predicted_price = data.get("predicted_price", 0.0)
                    
                    st.success("Target Successfully Inferenced!")
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1E2530 0%, #0E1117 100%); 
                                border: 2px solid #00C9FF; border-radius: 12px; padding: 30px; text-align: center;">
                        <h2 style="color: #A0AEC0; margin: 0; font-size: 1.5rem;">Estimated Trading Value</h2>
                        <h1 style="color: #92FE9D; margin: 10px 0 0 0; font-size: 4rem;">${predicted_price:,.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error(f"API Rejected Payload: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 CRITICAL: FastAPI Backend Offline. Execute `uvicorn backend_api:app` securely in Terminal.")
            except Exception as e:
                st.error(f"Prediction Pipeline Crash: {e}")

# =========================================================
# TAB 3: Groq AI Analyst (RAG LLM Connection)
# =========================================================
with tab_chat:
    st.markdown("### Semantic Market Analyst")
    st.markdown("Interact directly with the local ChromaDB vector store backed by Llama 3 Native Intelligence.")
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome back! How can I assist you with analyzing smartphone specifications today?"}
        ]
        
    # Render visible Chat Memory gracefully
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Handle Suggested Queries
    suggested_query = None
    if len(st.session_state.messages) == 1:
        st.markdown("<br>#### 💡 Suggested Queries", unsafe_allow_html=True)
        sq_col1, sq_col2, sq_col3 = st.columns(3)
        if sq_col1.button("What is the best phone under $500?"):
            suggested_query = "What is the best phone under $500?"
        if sq_col2.button("Compare Apple and Samsung battery life"):
            suggested_query = "Compare Apple and Samsung battery life"
        if sq_col3.button("Which phone has the most RAM?"):
            suggested_query = "Which phone has the most RAM?"
            
        st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Input Bar
    chat_input_val = st.chat_input("Ex: 'How much RAM is in the Samsung Galaxy S24 Ultra?'")
    user_prompt = chat_input_val if chat_input_val else suggested_query
    
    if user_prompt:
        
        # Append and display User String
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Searching semantic dimensions..."):
                try:
                    # Issue Decoupled LCEL POST natively 
                    response = requests.post(
                        f"{API_BASE_URL}/chat", 
                        json={"query": user_prompt}, 
                        timeout=30 # LLMs require extended timeouts sequentially
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        ai_reply = data.get("response", "Error locating AI Stream payloads.")
                        
                        # Graceful handling of missing contexts
                        if "do not have that data" in ai_reply.lower() or "not in the context" in ai_reply.lower():
                            safe_reply = "I apologize, but I only have access to information within the verified smartphone dataset up to the current cutoff date. I cannot definitively verify those specific hardware parameters."
                            message_placeholder.warning(safe_reply)
                            st.session_state.messages.append({"role": "assistant", "content": safe_reply})
                        else:
                            # Apply typewriter styling streaming effect
                            full_response = ""
                            for chunk in ai_reply.split():
                                full_response += chunk + " "
                                time.sleep(0.02)
                                message_placeholder.markdown(full_response + "▌")
                            
                            message_placeholder.markdown(full_response)
                            
                            # Save native context sequentially
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                    else:
                        error_msg = f"Server Execution Failed: {response.text}"
                        message_placeholder.error(error_msg)
                        
                except requests.exceptions.ConnectionError:
                    message_placeholder.error("🚨 CRITICAL: FastAPI Core Engine Disconnected. Launch `uvicorn backend_api:app`.")
                except requests.exceptions.Timeout:
                    message_placeholder.error("⏳ Inference Timeout: Llama 3 failed to return context quickly enough.")
                except Exception as e:
                    message_placeholder.error(f"UI Crash Context: {e}")
