# 0002: Versioned Qdrant collections; index state derived from Qdrant

Status: accepted (M2)

## Decision
- Collection name = `{prefix}__{model-slug}__d{dimension}__v{payload-schema}`,
  e.g. `portfolio_chunks__all-minilm-l6-v2__d384__v1`. Changing the model or the
  dimension selects a new, empty collection. Vectors from two models never share
  one. An existing collection with the wrong size or distance is refused
  (`CollectionMismatchError`).
- The ingestion registry (`QdrantDocumentRegistry`) is rebuilt from the point
  payloads (document id, content hash, chunker fingerprint, chunk ids). There is
  no local state file, so ingestion run from any machine syncs whichever Qdrant
  instance it points at.
- Point id = chunk id (UUIDv5), so an upsert overwrites and never duplicates.
- Every chunk is embedded before anything is written. A model failure therefore
  leaves the collection untouched.
- Searches always filter `visibility = public`.

## Migration (model change)
1. Set `PORTFOLIO_AI_EMBEDDING_MODEL` / `_DIMENSION` to the new model.
2. `portfolio-ai-ingest`: fills the new collection.
3. Verify with `portfolio-ai-search`, then delete the old collection.

## Deployment
Local Docker Qdrant is for development. Production (M9) can use Qdrant Cloud:
set an https `PORTFOLIO_AI_QDRANT_URL` and `PORTFOLIO_AI_QDRANT_API_KEY` (remote
plain http or a missing key is refused), then run `portfolio-ai-ingest` against
it. Chunks indexed locally do not appear there automatically.
