"""Deterministic scope classification: relevant, irrelevant, ambiguous, follow-ups."""

import pytest

from portfolio_ai.rag.retrieval.base import SearchResult
from portfolio_ai.rag.scope import Scope, classify, classify_before_retrieval
from tests.unit.test_qdrant_store import WORK, chunks_of

EVIDENCE = [SearchResult(chunk=c, score=0.45) for c in chunks_of(WORK)]  # goCare, Tegal


@pytest.mark.parametrize(
    "question",
    [
        # Clearly about Irfan or his field
        "What projects has Irfan built?",
        "What is his experience with Flutter?",
        "Which projects use Firebase?",
        "Does he know Python and machine learning?",
        "What services do you offer for mobile apps?",
        "Has Irfan built healthcare or telemedicine apps?",
        "Which apps handle payments?",
        "Where did he study? What is his education?",
        "Is Irfan available for freelance work?",
        "What is his hourly rate?",  # relevant even though the portfolio has no answer
        "Can you build my app?",
        # Broad terms that refer to his work must not be rejected
        "What AI work has been done?",
        "Tell me about the software he develops",
        "What technologies are used in these projects?",
        # Off-topic wording, but explicitly about Irfan: answered from evidence, never invented
        "Tell me a joke about Irfan's projects",
    ],
)
def test_relevant_questions_are_in_scope(question: str) -> None:
    assert classify(question, has_history=False, results=[]) is Scope.IN_SCOPE


@pytest.mark.parametrize(
    "question",
    [
        "Who won the football World Cup?",
        "What is the weather today?",
        "Tell me a joke.",
        "Solve 3x + 5 = 20",
        "What is 245 * 17?",
        "Write an essay about the French revolution",
        "What is the capital of France?",
        "Give me a pasta recipe",
        "What's the best pizza topping?",
        "Who is the president of the United States?",
    ],
)
def test_irrelevant_questions_are_out_of_scope(question: str) -> None:
    assert classify(question, has_history=False, results=[]) is Scope.OUT_OF_SCOPE


@pytest.mark.parametrize(
    "question",
    [
        "Who won the World Cup?",
        "Tell me a joke",
        "What is the weather in Lahore?",
        "What is 2 + 2?",
    ],
)
def test_clear_general_requests_are_decided_before_retrieval(question: str) -> None:
    assert classify_before_retrieval(question) is Scope.OUT_OF_SCOPE


def test_ambiguous_question_with_matching_evidence_is_in_scope() -> None:
    """No pronoun, no domain word, but the evidence names it (a project name)."""
    assert classify("Tell me about goCare", has_history=False, results=EVIDENCE) is Scope.IN_SCOPE
    assert classify("Tell me about goCare", has_history=False, results=[]) is Scope.OUT_OF_SCOPE


def test_low_score_alone_does_not_reject_a_portfolio_question() -> None:
    # No evidence above the threshold at all, yet clearly about his work.
    assert (
        classify("Has he worked with Rust or Go?", has_history=False, results=[]) is Scope.IN_SCOPE
    )


@pytest.mark.parametrize(
    "question", ["Which one?", "What about that?", "And the other one?", "Tell me more"]
)
def test_follow_ups_are_in_scope_only_within_a_conversation(question: str) -> None:
    assert classify(question, has_history=True, results=[]) is Scope.IN_SCOPE


def test_off_topic_follow_up_is_still_out_of_scope() -> None:
    assert (
        classify("And who won the World Cup?", has_history=True, results=[]) is Scope.OUT_OF_SCOPE
    )
