"""Grounded question answering: retrieve -> select evidence -> generate -> verify.

Guarantees, independent of what the model does:
- Citations come from retrieved chunks only; model-written URLs are never used.
- No evidence above the score threshold -> a fixed abstention, and no LLM call.
- An answer that claims sufficiency but cites nothing is flagged as insufficient.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from portfolio_ai.rag.generation.base import GenerationRequest, LLMProvider, Message, Role
from portfolio_ai.rag.generation.prompts import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    SYSTEM_INSTRUCTION,
    EvidenceItem,
    render_user_turn,
    select_evidence,
)
from portfolio_ai.rag.ingestion.cleaning import EMAIL_PLACEHOLDER, PHONE_PLACEHOLDER
from portfolio_ai.rag.retrieval.base import SearchResult
from portfolio_ai.rag.retrieval.retriever import SemanticRetriever

logger = logging.getLogger(__name__)

_PLACEHOLDERS = re.compile(re.escape(EMAIL_PLACEHOLDER) + "|" + re.escape(PHONE_PLACEHOLDER))


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    section: str
    url: str


class ChatAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    citations: list[Citation]
    insufficient_evidence: bool


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


@dataclass(frozen=True)
class AnswerSettings:
    top_k: int = 6
    min_score: float = 0.30
    evidence_token_budget: int = 1_800
    history_turns: int = 3
    history_answer_chars: int = 600
    temperature: float | None = None
    max_output_tokens: int = 800


class AnswerService:
    def __init__(
        self,
        retriever: SemanticRetriever,
        llm: LLMProvider,
        settings: AnswerSettings | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings or AnswerSettings()

    def answer(self, question: str, history: Sequence[Turn] = ()) -> ChatAnswer:
        settings = self._settings
        history = list(history)[-settings.history_turns :]
        results = self._retrieve(question, history)
        evidence = select_evidence(results, token_budget=settings.evidence_token_budget)

        if not evidence:
            logger.info("answer_abstained", extra={"reason": "no_evidence_above_threshold"})
            return ChatAnswer(
                answer=INSUFFICIENT_EVIDENCE_ANSWER, citations=[], insufficient_evidence=True
            )

        request = GenerationRequest(
            system_instruction=SYSTEM_INSTRUCTION,
            messages=[*self._history_messages(history), self._user_message(question, evidence)],
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
        )
        generated = self._llm.generate_grounded(request)

        by_id = {item.source_id: item for item in evidence}
        cited = [by_id[sid] for sid in dict.fromkeys(generated.cited_sources) if sid in by_id]
        unknown = set(generated.cited_sources) - by_id.keys()
        insufficient = generated.insufficient_evidence or not cited
        if not generated.insufficient_evidence and not cited:
            logger.warning("answer_without_valid_citations", extra={"unknown_ids": sorted(unknown)})

        answer_text = _PLACEHOLDERS.sub("the Contact section", generated.answer).strip()
        if not answer_text:
            answer_text = INSUFFICIENT_EVIDENCE_ANSWER
            insufficient = True

        logger.info(
            "answer_generated",
            extra={
                "evidence": len(evidence),
                "cited": len(cited),
                "unknown_citations": len(unknown),
                "insufficient_evidence": insufficient,
                "best_score": round(results[0].score, 4),
            },
        )
        return ChatAnswer(
            answer=answer_text,
            citations=_citations(cited),
            insufficient_evidence=insufficient,
        )

    def _retrieve(self, question: str, history: Sequence[Turn]) -> list[SearchResult]:
        """Search the question alone and, for follow-ups, with the previous question.

        "Which of those use Firebase?" means little alone; prefixing the previous
        question restores the referent. Results are merged keeping the best score.
        """
        settings = self._settings
        queries = [question]
        if history:
            queries.append(f"{history[-1].question} {question}")
        best: dict[str, SearchResult] = {}
        for query in queries:
            for result in self._retriever.search(
                query, top_k=settings.top_k, score_threshold=settings.min_score
            ):
                current = best.get(result.chunk.chunk_id)
                if current is None or result.score > current.score:
                    best[result.chunk.chunk_id] = result
        return sorted(best.values(), key=lambda r: r.score, reverse=True)[: settings.top_k]

    def _history_messages(self, history: Sequence[Turn]) -> list[Message]:
        limit = self._settings.history_answer_chars
        messages: list[Message] = []
        for turn in history:
            messages.append(Message(role=Role.USER, text=turn.question))
            messages.append(Message(role=Role.MODEL, text=turn.answer[:limit]))
        return messages

    @staticmethod
    def _user_message(question: str, evidence: Sequence[EvidenceItem]) -> Message:
        return Message(role=Role.USER, text=render_user_turn(question, evidence))


def _citations(items: Sequence[EvidenceItem]) -> list[Citation]:
    citations: dict[tuple[str, str], Citation] = {}
    for item in items:
        chunk = item.result.chunk
        key = (chunk.source_uri, chunk.section)
        citations.setdefault(
            key, Citation(title=chunk.document_title, section=chunk.section, url=chunk.source_uri)
        )
    return list(citations.values())
