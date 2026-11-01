"""Framework-independent RAG engine.

Rule: nothing in this package may import FastAPI or `portfolio_ai.api`.
API routes depend on services; services depend on `rag`; never the reverse.
"""
