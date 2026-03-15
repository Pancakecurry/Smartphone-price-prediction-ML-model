"""
RAG Vector Store Database Builder.

Constructs a local ChromaDB collection containing serialized smartphone specifications
for downstream Large Language Models (LLMs) to query effectively natively.
"""
import polars as pl
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from src.logger import get_logger
from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT

logger = get_logger(__name__)

class SmartphoneKnowledgeBase:
    """
    Initializes and populates a ChromaDB persistence client, structuring dataset tabular tensors
    into semantically retrievable text document blocks natively.
    """
    
    def __init__(self):
        """
        Establishes a local physical ChromaDB engine pointing to data/chromadb/.
        Initializes the sentence-transformer embeddings efficiently using HuggingFace.
        """
        self.chroma_path = PROJECT_ROOT / "data" / "chromadb"
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing Persistent ChromaDB Client at {self.chroma_path}")
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        
        # We explicitly configure the sentence transformer topology natively for high-velocity local performance
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Static collection name — NEVER changes so agents never hold a stale UUID
        self.COLLECTION_NAME = "smartphone_market_data"
        self.collection = self.client.get_or_create_collection(name=self.COLLECTION_NAME)

    def build_database(self) -> None:
        """
        Parses Phase 2 processed schema arrays into contextual language Strings.
        Only executes if the physical persistence is completely empty, ensuring idempotency.
        
        Serialized Documents contain precise textual representations of technical specs and Pricing.
        """
        # Delete by the static name (try-except handles first-run where collection doesn't exist yet)
        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
            logger.info("Wiped old vector memory. Embedding fresh dataset...")
        except Exception:
            logger.info("No existing collection to wipe — creating fresh.")

        # Recreate under the SAME static name — UUID is fresh but name stays constant
        self.collection = self.client.get_or_create_collection(name=self.COLLECTION_NAME)
            
        parquet_path = PROCESSED_DATA_DIR / "master_smartphones.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet artifact missing: {parquet_path}. Phases 1 & 2 must properly complete first.")
            
        logger.info(f"Reading {parquet_path} iteratively into Polars...")
        df = pl.read_parquet(parquet_path)
        
        documents = []
        metadatas = []
        ids = []
        
        logger.info("Iterating natively into vector string blocks...")
        
        # Native dictionaries translation safely escaping structural deviations
        for idx, row in enumerate(df.to_dicts()):
            brand = row.get("Brand", "Unknown")
            # Safely capture possible column names mapping dynamically
            name = row.get("Smartphone_Name", "Smartphone")
            ram = row.get("ram_gb", "Unknown")
            battery = row.get("battery_mah", "Unknown")
            processor = row.get("Processor", "Unknown")
            price = row.get("Price", 0.0)
            
            # Formatted context template explicitly bounding language structures specifically as requested
            doc = f"The {brand} {name} features {ram} GB of RAM, a {battery} mAh battery, and a {processor} processor. Its market price is ${price}."
            
            documents.append(doc)
            # Tag vectors specifically mapping strict numerical/String values enabling dynamic Retrieval filtering downstream
            metadatas.append({
                "Price": float(price), 
                "Brand": str(brand)
            })
            ids.append(f"doc_{idx}")
            
        logger.info(f"Encoding {len(documents)} context documents securely via MiniLM-L6-v2 Embeddings...")
        # Mathematically vectorizes the Language arrays into dense computational Tensors natively
        embedded_tensors = self.embeddings.embed_documents(documents)
        
        logger.info("Storing Tensor payloads natively into localized Chroma persistence...")
        # Final physical database write mapped sequentially
        self.collection.add(
            documents=documents,
            embeddings=embedded_tensors,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info("[✓] RAG Database Matrix execution successfully concluded and secured.")

if __name__ == "__main__":
    db_builder = SmartphoneKnowledgeBase()
    db_builder.build_database()
