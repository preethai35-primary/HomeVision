"""
agent/tracing.py — shared LangFuse observability client
Phase: 1 (set up now, used throughout)

Import langfuse from here everywhere — never instantiate Langfuse() twice.
"""
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

# single shared client — reads LANGFUSE_* keys from .env automatically
# if keys are missing, LangFuse silently no-ops (tracing is optional)
try:
    langfuse = Langfuse()
except Exception:
    # if langfuse not installed or keys missing, use a no-op stub
    class _NoOpLangfuse:
        def trace(self, **kwargs): return self
        def span(self, **kwargs): return self
        def generation(self, **kwargs): return self
        def update(self, **kwargs): return self
        def end(self, **kwargs): return self
        def flush(self): pass
    langfuse = _NoOpLangfuse()


def get_callback_handler():
    """LangChain/LangGraph callback — pass to agent.invoke() config for auto-tracing."""
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler()
    except Exception:
        return None
