import streamlit as st

def inject_custom_css():
    st.markdown("""
    <style>
        /* Modern Apple-Inspired Font Stack */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif !important;
        }

        /* Hide Default UI */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(128, 128, 128, 0.08);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.04);
            margin-bottom: 24px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .glass-card:hover {
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }

        /* Metric Cards */
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(135deg, var(--primary-color, #2563EB) 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .metric-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #8E8E93;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }
        
        /* Chat UI styling overrides */
        div[data-testid="stChatMessage"] {
            background: rgba(128, 128, 128, 0.05);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 16px;
            padding: 16px;
            margin: 12px 0;
            box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        }
        
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: rgba(37, 99, 235, 0.08);
            border: 1px solid rgba(37, 99, 235, 0.2);
            border-radius: 16px;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.2s ease;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            border-color: var(--primary-color, #2563EB);
            color: var(--primary-color, #2563EB);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: rgba(128, 128, 128, 0.02);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(128, 128, 128, 0.1);
        }
        
        /* Inputs & Selects */
        .stSelectbox>div>div, .stNumberInput>div>div, .stTextInput>div>div {
            border-radius: 10px !important;
        }
        
        /* Typography overrides */
        h1, h2, h3, h4 {
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
