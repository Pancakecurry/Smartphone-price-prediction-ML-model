import streamlit as st

def inject_custom_css():
    st.markdown("""
<style>
/* 1. IMPORT PREMIUM TYPOGRAPHY */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* 2. SUBTLE ANIMATED BACKGROUND (Dark Premium Mesh) */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #0f172a, #334155) !important;
    background-size: 400% 400% !important;
    animation: gradientBG 15s ease infinite !important;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* 3. SLEEK CUSTOM SCROLLBARS */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
}

/* 4. BUTTON MICRO-INTERACTIONS & GRADIENTS */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 15px rgba(168, 85, 247, 0.4) !important;
}

/* 5. SIDEBAR GLASSMORPHISM ENHANCEMENT */
[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* 6. KEEP THE HEADER TRANSPARENT & TOGGLE VISIBLE */
header { 
    background-color: transparent !important; 
}
[data-testid="collapsedControl"] {
    display: flex !important;
    z-index: 999999;
}
.stDeployButton { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }

/* 7. SOFTEN INPUT BOXES */
.stTextInput > div > div > input, 
.stNumberInput > div > div > input {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: white !important;
}
.stTextInput > div > div > input:focus, 
.stNumberInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}
</style>
    """, unsafe_allow_html=True)
