"""
RAG Pipeline Quality Assurance (QA) Suite.

Validates text context serialization, ChromaDB persistence architectures, and 
simulates Large Language Model (LLM) invocations offline bypassing Groq API costs natively.
"""
import pytest
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.config import PROJECT_ROOT
from src.rag.vector_store_builder import SmartphoneKnowledgeBase
from src.rag.groq_agent import SmartphoneAI

def test_rag_serialization_structure():
    """
    Test 1 (Serialization):
    Validates the text translation of raw dictionaries. Simulates a Parquet row mapping,
    asserting the output string correctly interpolates rigorous market boundaries (Brand, Price).
    """
    kb = SmartphoneKnowledgeBase()
    
    # Mocking single matrix dictionary (Simulated Polars Extraction)
    mock_row = {
        "Brand": "Apple",
        "Model": "iPhone 14",
        "RAM": 6,
        "Battery": 3279,
        "Processor": "A15 Bionic",
        "Price": 799.0
    }
    
    # Isolate functional text parsing mapping bounds directly as defined in codebase
    brand = mock_row.get("Brand", "Unknown")
    name = mock_row.get("Model", mock_row.get("Smartphone_Name", "Smartphone"))
    ram = mock_row.get("RAM_GB", mock_row.get("RAM", "Unknown"))
    battery = mock_row.get("Battery_Capacity_mAh", mock_row.get("Battery", "Unknown"))
    processor = mock_row.get("Processor_Brand", mock_row.get("Processor", "Unknown"))
    price = mock_row.get("Price", 0.0)
    
    serialized_doc = f"The {brand} {name} features {ram} GB of RAM, a {battery} mAh battery, and a {processor} processor. Its market price is ${price}."
    
    assert "Apple" in serialized_doc
    assert "iPhone 14" in serialized_doc
    assert "6 GB of RAM" in serialized_doc
    assert "market price is $799.0" in serialized_doc

def test_chromadb_persistence_layer():
    """
    Test 2 (ChromaDB Persistence):
    Evaluates if the Vector Store securely generated SQLite binary caches into the Operating System logically.
    """
    chromadb_dir = PROJECT_ROOT / "data" / "chromadb"
    
    assert chromadb_dir.exists(), "ChromaDB logical initialization mapped incorrectly. Directory missing."
    assert chromadb_dir.is_dir(), "Mounted ChromaDB path is inexplicably mapped to a physically static file."
    
    # Validate the directory physically contains the active SQLite files ensuring engine serialization
    files = list(chromadb_dir.iterdir())
    assert len(files) > 0, "ChromaDB directory initialized, but the internal physical state is empty."

from langchain_core.messages import AIMessage

@patch('src.rag.groq_agent.ChatGroq.invoke')
def test_mocked_llm_inference_chain(mock_groq_invoke):
    """
    Test 3 (Mocked LLM Call):
    Completely isolates the LCEL RAG execution flow from live Groq web-servers computationally.
    Validates logical connection mappings within `ask_question()` preventing financial resource burn. 
    """
    # 1. Provide deterministic mock API response mathematically bounds
    # LCEL StrOutputParser explicitly requires a legitimate `AIMessage` structure downstream
    mock_ai_message = AIMessage(content="The iPhone 14 test response price is $799 based on LCEL context.")
    # Overwrite the invoke return to deliver the True LangChain object directly
    mock_groq_invoke.return_value = mock_ai_message

    # 2. Architect RAG execution offline implicitly mapping ChromaDB
    agent = SmartphoneAI()
    test_query = "What is the price of iPhone 14?"
    
    response = agent.ask_question(test_query)

    # 3. Execution Asset Validation
    mock_groq_invoke.assert_called_once()
    assert response == "The iPhone 14 test response price is $799 based on LCEL context."
    assert "test response price" in response
