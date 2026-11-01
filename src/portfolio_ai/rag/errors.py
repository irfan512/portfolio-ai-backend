"""Exception hierarchy for the RAG engine.

Callers catch the narrowest class they can handle. The ingestion pipeline catches
`IngestionError` per source, so one broken source never aborts the others.
"""


class RagError(Exception):
    """Base class for every error raised by `portfolio_ai.rag`."""


class IngestionError(RagError):
    """A source could not be ingested."""


class UnapprovedSourceError(IngestionError):
    """A URL is outside the approved source policy (wrong host, scheme or path)."""


class FetchError(IngestionError):
    """A source could not be downloaded or rendered."""


class ParseError(IngestionError):
    """A source was fetched but yielded no usable content."""


class EmbeddingError(RagError):
    """The embedding model failed, is unreachable, or returned unusable vectors."""


class EmbeddingDimensionError(EmbeddingError):
    """Vectors do not have the dimension the collection was built for."""


class VectorStoreError(RagError):
    """The vector database is unreachable or rejected an operation."""


class CollectionMismatchError(VectorStoreError):
    """An existing collection was built with different vector settings."""


class LLMError(RagError):
    """The language model could not produce a usable answer."""


class LLMConfigurationError(LLMError):
    """The model is not configured (missing key or model name)."""


class LLMTimeoutError(LLMError):
    """The model did not answer within the timeout."""


class LLMRateLimitError(LLMError):
    """The provider rejected the request because of quota or rate limits."""


class LLMUnavailableError(LLMError):
    """The provider failed (5xx, network) after retries."""


class LLMResponseError(LLMError):
    """The model answered, but not in the required structured format."""
