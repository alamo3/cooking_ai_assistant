"""ASGI entry point: `uvicorn cooking_assistant_ai.main:app`."""
from cooking_assistant_ai.api.app import create_app

app = create_app()
