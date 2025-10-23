# ObservingAI
The scope of this project is to establish repeatable, best practices for monitoring LLMs and AI for Obseravbility
# AI Observability Roadmap

- Month 3: Building ML pipelines with Kubeflow
- Learning: Model deployment on Azure ML
- Upcoming: LLM Observability with OpenTelemetry

- `/01_foundations` → ML & Python basics
- `/03_mlop_pipeline` → CI/CD + model deployment
- `/05_ai_observability` → Applying OpenTelemetry to AI workloads

Python • PyTorch • TensorFlow • Docker • Kubernetes • MLflow • OpenTelemetry • Azure ML • AWS SageMaker

Quick Run Script:
mkdir rag-chatbot; cd rag-chatbot
mkdir src, docs
New-Item -Path "src\__init__.py" -ItemType File
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install langchain==0.1.0 langchain-openai==0.0.5 langchain-community==0.0.13 chromadb==0.4.22 pypdf==4.0.0 opentelemetry-api==1.22.0 opentelemetry-sdk==1.22.0 opentelemetry-exporter-otlp-proto-grpc==1.22.0 opentelemetry-instrumentation-fastapi==0.43b0 openai==1.10.0 fastapi==0.109.0 uvicorn[standard]==0.27.0 python-dotenv==1.0.0 tiktoken==0.5.2 pydantic==2.5.0

Copy the 3 Python files from artifacts above
Create .env from the template and add your Azure OpenAI credentials
Run: python src/api.py
Test: curl http://localhost:8000/health
Open: http://localhost:8000/docs for interactive API