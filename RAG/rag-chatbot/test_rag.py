# test_rag.py
from src.rag_engine import ObservableRAG, RAGConfig

# Initialize
config = RAGConfig()
rag = ObservableRAG(config)

# Ingest documents
rag.ingest_documents(["docs/sample.txt"])

# Setup QA chain
rag.setup_qa_chain()

# Query
result = rag.query("What is OpenTelemetry?")
print(result['answer'])