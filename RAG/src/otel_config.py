from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Configure OTLP exporter
otlp_exporter = OTLPSpanExporter(
    endpoint="https://ingest.us1.signalfx.com:443",
    headers={"X-SF-Token": os.getenv("SPLUNK_ACCESS_TOKEN")}
)

trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))