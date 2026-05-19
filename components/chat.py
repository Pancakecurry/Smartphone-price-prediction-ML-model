import streamlit as st
import requests
import time

def render_chat_interface(api_base_url):
    st.markdown("## AI Assistant")
    st.markdown("<p style='color:#8E8E93;'>Interact directly with the local ChromaDB vector store backed by Llama 3.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome back! How can I assist you with analyzing smartphone specifications today?"}
        ]
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    suggested_query = None
    if len(st.session_state.messages) == 1:
        st.markdown("<br>#### 💡 Suggested Queries", unsafe_allow_html=True)
        sq_col1, sq_col2, sq_col3 = st.columns(3)
        if sq_col1.button("Best phone under $500?"):
            suggested_query = "What is the best phone under $500?"
        if sq_col2.button("Compare Apple & Samsung"):
            suggested_query = "Compare Apple and Samsung battery life"
        if sq_col3.button("Phone with most RAM?"):
            suggested_query = "Which phone has the most RAM?"
            
        st.markdown("<br>", unsafe_allow_html=True)
        
    chat_input_val = st.chat_input("Ex: 'How much RAM is in the Samsung Galaxy S24 Ultra?'")
    user_prompt = chat_input_val if chat_input_val else suggested_query
    
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Searching semantic dimensions..."):
                try:
                    response = requests.post(
                        f"{api_base_url}/chat", 
                        json={"query": user_prompt}, 
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        ai_reply = data.get("response", "Error locating AI Stream payloads.")
                        
                        if "do not have that data" in ai_reply.lower() or "not in the context" in ai_reply.lower():
                            safe_reply = "I apologize, but I only have access to information within the verified smartphone dataset up to the current cutoff date."
                            message_placeholder.warning(safe_reply)
                            st.session_state.messages.append({"role": "assistant", "content": safe_reply})
                        else:
                            full_response = ""
                            for chunk in ai_reply.split():
                                full_response += chunk + " "
                                time.sleep(0.02)
                                message_placeholder.markdown(full_response + "▌")
                            
                            message_placeholder.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                    else:
                        message_placeholder.error(f"Server Execution Failed: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    message_placeholder.error("🚨 CRITICAL: FastAPI Core Engine Disconnected.")
                except requests.exceptions.Timeout:
                    message_placeholder.error("⏳ Inference Timeout: Llama 3 failed to return context quickly enough.")
                except Exception as e:
                    message_placeholder.error(f"UI Crash Context: {e}")
