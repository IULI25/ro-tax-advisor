import streamlit as st
from arize.otel import register
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

_tracer_provider = None

def initializeaza_tracing():
    """Pornește instrumentarea Phoenix o singură dată per proces."""
    global _tracer_provider
    if _tracer_provider is not None:
        return _tracer_provider

    space_id = st.secrets.get("PHOENIX_SPACE_ID")
    api_key = st.secrets.get("PHOENIX_API_KEY")

    if not space_id or not api_key:
        # fără chei Phoenix -> aplicația funcționează normal, doar fără tracing
        return None

    _tracer_provider = register(
        space_id=space_id,
        api_key=api_key,
        project_name="consilier-ai-legislativ",
    )

    GoogleGenAIInstrumentor().instrument(tracer_provider=_tracer_provider)
    return _tracer_provider

def obtine_tracer():
    from opentelemetry import trace
    return trace.get_tracer(__name__)