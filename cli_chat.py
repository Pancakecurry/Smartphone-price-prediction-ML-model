"""
Smartphone AI Command Line Interface (CLI).

A lightweight, interactive terminal wrapper allowing live RAG inference testing 
against the Llama-3 LangChain logic securely before deploying physical Web UIs.
"""
import sys
from src.rag.groq_agent import SmartphoneAI
from src.logger import get_logger

logger = get_logger(__name__)

def run_interactive_cli():
    """
    Mounts a continuous loop capturing User Standard Input (stdin), 
    streaming semantic logic natively into ChatGroq and returning grounded outputs.
    """
    try:
        print("\n" + "=" * 60)
        print("🤖 Initialization: Mounting Smartphone AI (Llama 3 Core)...")
        agent = SmartphoneAI()
        print("✅ Vector Database & Inference Endpoints Connected.")
        print("=" * 60)
    except Exception as e:
        logger.critical(f"Failed to bootstrap SmartphoneAI: {e}")
        print(f"\n[CRITICAL ERROR] Failed to mount AI Model: {e}")
        sys.exit(1)

    while True:
        try:
            # Capture Input
            user_query = input('\nAsk the Smartphone AI (or type "exit" to quit): ').strip()
            
            # Halting condition
            if user_query.lower() in ["exit", "quit", "q"]:
                print("\n[System] Powering down Smartphone AI logic. Goodbye! 👋\n")
                break
                
            if not user_query:
                print("⚠️ [Warning] Cannot query empty text strings. Please explicitly ask a question.")
                continue

            print("\n" + "-" * 40)
            print("🧠 Searching Knowledge Base & Streaming Thought...")
            
            # Submitting query synchronously bypassing timeout errors organically
            try:
                ai_response = agent.ask_question(user_query)
            except Exception as api_err:
                print(f"[API ERROR] Network logic failed during LLM invocation: {api_err}")
                print("-" * 40)
                continue
                
            # Error Fallback Formatting
            if not ai_response or "Exception" in ai_response:
                print("❌ [Engine Failure] Unable to connect computationally or retrieve database vectors.")
                print(f"Details: {ai_response}")
            else:
                print(f"\n📱 AI RESPONSE:\n{ai_response}")
            print("-" * 40)

        # Handling explicit CTRL+C interruptions securely
        except KeyboardInterrupt:
            print("\n\n[System] Keyboard Interrupt Detected. Safely terminating inference engine. 🛑\n")
            break
        except Exception as iter_e:
            print(f"\n[System Error] Unhandled Exception occurred during conversation loop: {iter_e}\n")

if __name__ == "__main__":
    run_interactive_cli()
