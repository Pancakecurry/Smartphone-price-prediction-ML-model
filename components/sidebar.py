import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; font-weight: 800; letter-spacing: -0.05em;'>📱 Market AI</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 🧭 Navigation")
        selected_page = st.radio(
            "Go to:",
            [
                "📈 Market Analytics", 
                "⚖️ Smartphone Comparison", 
                "🤖 AI Assistant", 
                "🔄 Sync Live Data", 
                "⚙️ System Settings"
            ],
            label_visibility="collapsed" 
        )
        
        st.markdown("---")
        return selected_page
