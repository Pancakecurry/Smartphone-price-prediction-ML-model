from src.rag.groq_agent import SmartphoneAI

def test_retriever():
    agent = SmartphoneAI()
    print("\n--- Testing Raw Chroma Retrieval ---")
    query = "how much ram is in samsung s24 ultra ?"
    print(f"Query: {query}\n")
    
    docs = agent.retriever.invoke(query)
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.page_content}")

if __name__ == "__main__":
    test_retriever()
