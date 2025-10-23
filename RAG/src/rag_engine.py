"""
RAG Chatbot with OpenTelemetry Observability
A production-ready RAG implementation with full observability
"""

import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass

# Core RAG dependencies
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode

# Initialize OpenTelemetry
resource = Resource(attributes={
    "service.name": "rag-chatbot",
    "service.version": "1.0.0",
    "deployment.environment": "development"
})

# Setup Tracing
trace_provider = TracerProvider(resource=resource)
trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)

# Setup Metrics
metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=5000)
metric_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(metric_provider)
meter = metrics.get_meter(__name__)

# Define metrics
query_counter = meter.create_counter(
    "rag.queries.total",
    description="Total number of RAG queries"
)

query_duration = meter.create_histogram(
    "rag.query.duration",
    description="RAG query duration in seconds",
    unit="s"
)

token_counter = meter.create_counter(
    "rag.tokens.total",
    description="Total tokens used",
    unit="tokens"
)

retrieval_counter = meter.create_counter(
    "rag.documents.retrieved",
    description="Number of documents retrieved"
)


@dataclass
class RAGConfig:
    """Configuration for RAG chatbot"""
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_key: str = os.getenv("AZURE_OPENAI_KEY", "")
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    azure_openai_api_version: str = "2024-02-15-preview"
    embedding_deployment: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 4
    temperature: float = 0.7
    vector_db_path: str = "./chroma_db"


class ObservableRAG:
    """RAG Chatbot with built-in observability"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.azure_openai_endpoint,
            azure_deployment=config.azure_openai_deployment,
            api_version=config.azure_openai_api_version,
            api_key=config.azure_openai_key,
            temperature=config.temperature
        )
        
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=config.azure_openai_endpoint,
            azure_deployment=config.embedding_deployment,
            api_key=config.azure_openai_key
        )
    
    @tracer.start_as_current_span("ingest_documents")
    def ingest_documents(self, file_paths: List[str]):
        """Ingest documents into vector store with tracing"""
        span = trace.get_current_span()
        span.set_attribute("document.count", len(file_paths))
        
        documents = []
        for file_path in file_paths:
            try:
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Failed to load {file_path}"))
                print(f"Error loading {file_path}: {e}")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        span.set_attribute("chunks.count", len(splits))
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.config.vector_db_path
        )
        
        print(f"✅ Ingested {len(documents)} documents into {len(splits)} chunks")
        return len(splits)
    
    @tracer.start_as_current_span("setup_qa_chain")
    def setup_qa_chain(self):
        """Setup the QA chain with custom prompt"""
        
        prompt_template = """You are a helpful AI assistant specializing in Splunk and observability.
Use the following context to answer the question. If you don't know the answer, say so.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.top_k_results}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("✅ QA Chain configured")
    
    @tracer.start_as_current_span("query")
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with full observability"""
        start_time = time.time()
        span = trace.get_current_span()
        span.set_attribute("query.text", question)
        
        # Increment query counter
        query_counter.add(1, {"status": "started"})
        
        try:
            # Execute query
            result = self.qa_chain.invoke({"query": question})
            
            # Record metrics
            duration = time.time() - start_time
            query_duration.record(duration)
            
            # Track retrieved documents
            num_docs = len(result.get("source_documents", []))
            retrieval_counter.add(num_docs)
            span.set_attribute("documents.retrieved", num_docs)
            span.set_attribute("query.duration_seconds", duration)
            
            # Estimate tokens (rough estimate: 1 token ≈ 4 chars)
            total_chars = len(question) + len(result.get("result", ""))
            estimated_tokens = total_chars // 4
            token_counter.add(estimated_tokens, {"type": "estimated"})
            span.set_attribute("tokens.estimated", estimated_tokens)
            
            query_counter.add(1, {"status": "success"})
            span.set_status(Status(StatusCode.OK))
            
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "duration_seconds": duration,
                "num_sources": num_docs
            }
            
        except Exception as e:
            query_counter.add(1, {"status": "error"})
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def main():
    """Example usage"""
    
    # Configuration
    config = RAGConfig()
    
    # Initialize RAG
    rag = ObservableRAG(config)
    
    # Example: Ingest sample documents
    # rag.ingest_documents(["./docs/splunk_guide.txt", "./docs/otel_guide.pdf"])
    
    # Load existing vector store
    rag.vectorstore = Chroma(
        persist_directory=config.vector_db_path,
        embedding_function=rag.embeddings
    )
    
    # Setup QA chain
    rag.setup_qa_chain()
    
    # Example queries
    questions = [
        "How do I configure OpenTelemetry with Splunk?",
        "What are the best practices for Kubernetes observability?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        result = rag.query(question)
        print(f"✅ Answer: {result['answer'][:200]}...")
        print(f"📊 Retrieved {result['num_sources']} sources in {result['duration_seconds']:.2f}s")


if __name__ == "__main__":
    main()