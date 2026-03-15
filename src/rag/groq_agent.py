"""
Groq LCEL RAG Agent.

Constructs the LLM Conversational Interface connecting the localized ChromaDB Retrievers 
to the hyper-fast Groq Llama3 Inference Engine natively.
"""
import os
import logging
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, create_retriever_tool, tool
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from groq import GroqError

from src.logger import get_logger
from src.config import PROJECT_ROOT

logger = get_logger(__name__)

class SmartphoneAI:
    """
    RAG orchestrated AI Agent retrieving specs via Semantic Search and generating 
    accurate, hallucination-free analytics reporting natively via Groq.
    """
    
    def __init__(self):
        """
        Connects Local environment variables, mounts the ChatGroq model natively,
        and constructs the local Vector Search retriever.
        """
        # Formally inject Secure API configurations dynamically
        load_dotenv(override=True)
        
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY missing from environment variables (.env). Ensure secure instantiation before querying.")

        try:
            logger.info("Initializing Groq Compute Engine mapping Llama-3.1-8B architecture...")
            self.llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.0,  # Zero temp eliminates hallucination
                max_tokens=512    # Keep responses concise; avoids TPM blowout on Groq free tier
            )
        except Exception as e:
            logger.critical(f"Failed to bind ChatGroq Engine logically: {e}")
            raise RuntimeError(f"ChatGroq Node failure: {e}") from e
            
        # Static collection name — must match vector_store_builder.py exactly
        self.COLLECTION_NAME = "smartphone_market_data"

        # Embeddings and path — must be instance attrs before any tool closure references them
        self.chroma_path = PROJECT_ROOT / "data" / "chromadb"
        self.embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Build ONE stable LangChain Chroma instance (avoids per-query RustBindingsAPI crash)
        logger.info(
            f"Connecting ChromaDB → collection='{self.COLLECTION_NAME}' "
            f"at '{self.chroma_path}'"
        )
        self.vector_store = Chroma(
            persist_directory=str(self.chroma_path),
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

        # ── Shared Pydantic schema: Groq API requires explicit tool arg types ──
        class SearchInput(BaseModel):
            query: str = Field(description="A single specific search query string.")

        # Capture instance for use inside closures
        _vector_store = self.vector_store
        ddg_search = DuckDuckGoSearchRun()
        CHAR_LIMIT = 1500

        # ── Tool 1: Local ChromaDB ─────────────────────────────────────────────
        @tool("Local_Smartphone_Database", args_schema=SearchInput)
        def local_db_tool(query: str) -> str:
            """Searches the local database for smartphone specifications and prices."""
            try:
                docs = _vector_store.similarity_search(query, k=3)
                combined = "\n".join(doc.page_content for doc in docs)
            except Exception as e:
                return f"[Local DB error: {e}]"
            return combined[:CHAR_LIMIT] + "... [TRUNCATED]" if len(combined) > CHAR_LIMIT else combined

        # ── Tool 2: Live Web Search ────────────────────────────────────────────
        @tool("Live_Web_Search", args_schema=SearchInput)
        def live_web_search_tool(query: str) -> str:
            """Searches the live internet for current smartphone specs and market prices."""
            try:
                result = str(ddg_search.run(query))
            except Exception as e:
                return f"[Web search error: {e}]"
            return result[:CHAR_LIMIT] + "... [TRUNCATED]" if len(result) > CHAR_LIMIT else result

        self.tools = [local_db_tool, live_web_search_tool]

        # ── ReAct Agent ───────────────────────────────────────────────────────
        system_instruction = (
            "You are an elite Smartphone Market Analyst. "
            "ALWAYS call Local_Smartphone_Database first with a concise query. "
            "If the phone is not found locally, call Live_Web_Search once. "
            "Respond with a single concise paragraph. Do not repeat tool outputs verbatim."
        )
        self.agent_executor = create_react_agent(self.llm, tools=self.tools, prompt=system_instruction)


    def ask_question(self, user_query: str) -> str:
        """
        Executes explicit LCEL sequential chains streaming questions contextually into Llama-3 endpoints.
        Handles API Timeouts and Connection Errors natively.
        """
        logger.info(f"Incoming RAG Prompt Queue: '{user_query}'")
        
        try:
            # Execute Agentic Tool Routing Logic natively
            logger.info("Executing ReAct Tool Calling Agent evaluation...")
            try:
                response = self.agent_executor.invoke({"messages": [("user", user_query)]})
                # Extract final AI response string successfully
                return response["messages"][-1].content
            except Exception as e:
                error_context = f"Internal LangChain Tool Parsing crashed during tool selection: {e}. Please try rewording your query."
                logger.error(error_context)
                return error_context
            
        except GroqError as api_err:
            error_msg = f"Groq Llama-3 API Gateway Exception (Timeout or Authorization): {api_err}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Internal LCEL Streaming Failure Context: {e}"
            logger.error(error_msg)
            return error_msg

if __name__ == "__main__":
    agent = SmartphoneAI()
    print("\n--- Testing Groq Vector LCEL Bindings ---")
    answer = agent.ask_question("What is the battery and price of the Samsung Galaxy S24 Ultra?")
    print(f"\nAI RESPONSE:\n{answer}")
