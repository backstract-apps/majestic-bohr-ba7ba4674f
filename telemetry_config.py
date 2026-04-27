import logging
import os
import sys

from loguru import logger
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource


def setup_telemetry_and_logging():
    # Environment-aware OpenTelemetry Configuration
    otel_service_name = os.getenv("OTEL_SERVICE_NAME", "backstract-dev")
    otel_exporter_otlp_endpoint = "http://localhost:4317"
    # Determine if the OTLP endpoint for logs should be insecure based on an env var or default
    # For local development, insecure=True is common. For production, it should typically be False unless it's internal traffic.
    otel_logs_exporter_insecure_str = "true"
    otel_logs_exporter_insecure = otel_logs_exporter_insecure_str == "true"

    # Configure resource with service name
    resource = Resource.create({"service.name": otel_service_name})

    # Set up trace provider
    # tracer_provider = TracerProvider(resource=resource)
    # trace.set_tracer_provider(tracer_provider)
    # otlp_trace_exporter = OTLPSpanExporter(
    #    endpoint=otel_exporter_otlp_endpoint)
    # span_processor = BatchSpanProcessor(otlp_trace_exporter)
    # tracer_provider.add_span_processor(span_processor)

    # Set up log provider
    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)
    otlp_log_exporter = OTLPLogExporter(
        endpoint=otel_exporter_otlp_endpoint, insecure=otel_logs_exporter_insecure
    )
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(otlp_log_exporter))

    # Instrument Python standard logging
    LoggingInstrumentor().instrument(set_logging_format=True)

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while (
                frame and frame.f_code.co_filename == logging.__file__
            ):  # Added 'frame and' for robustness
                frame = frame.f_back
                depth += 1

            # Extract OTel attributes and prepare for Loguru's extra
            log_extra = {
                "otelTraceID": getattr(record, "otelTraceID", None),
                "otelSpanID": getattr(record, "otelSpanID", None),
                "otelServiceName": getattr(record, "otelServiceName", None),
                # You can add otelTraceFlags if needed by uncommenting the next line
                # "otelTraceFlags": getattr(record, "otelTraceFlags", None),
            }
            # Filter out None values if you don't want them in 'extra' when logging
            log_extra_filtered = {k: v for k, v in log_extra.items() if v is not None}

            logger.opt(depth=depth, exception=record.exc_info).bind(
                **log_extra_filtered
            ).log(level, record.getMessage())

    # Configure loguru
    logger.remove()
    # Example of how to include trace context in Loguru's format:
    # logger.add(
    #     sys.stdout,
    #     format="{time} | {level} | TraceID: {extra[otelTraceID]} | SpanID: {extra[otelSpanID]} | {message}",
    #     serialize=False,
    #     backtrace=True,
    #     diagnose=True,
    # )
    # Current configuration (without trace context in format string):
    logger.add(
        sys.stdout,
        format="{time} | {level} | {message}",
        serialize=False,  # Plain text format is easier to parse
        backtrace=True,
        diagnose=True,
    )

    # Configure standard logging to use Loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG)
