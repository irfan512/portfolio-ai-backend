import pytest
from qdrant_client import QdrantClient

from portfolio_ai.rag.errors import LLMTimeoutError
from portfolio_ai.rag.generation.base import GroundedAnswer, Role
from portfolio_ai.rag.generation.prompts import INSUFFICIENT_EVIDENCE_ANSWER
from portfolio_ai.rag.retrieval.qdrant_store import QdrantChunkStore
from portfolio_ai.rag.retrieval.retriever import SemanticRetriever
from portfolio_ai.rag.service import AnswerService, AnswerSettings, Turn
from tests.fakes import FakeLLM, HashingEmbeddingProvider
from tests.unit.test_qdrant_store import EXPERIENCE, WORK, chunks_of

SITE = "https://irfan512.github.io/portfolio/"


@pytest.fixture
def retriever() -> SemanticRetriever:
    embedder = HashingEmbeddingProvider()
    store = QdrantChunkStore(QdrantClient(":memory:"), embedder, prefix="t")
    store.ensure_collection()
    store.upsert(chunks_of(WORK, EXPERIENCE))
    return SemanticRetriever(embedder, store)


def service(retriever: SemanticRetriever, llm: FakeLLM, **settings) -> AnswerService:
    return AnswerService(retriever, llm, AnswerSettings(min_score=0.1, **settings))


def test_grounded_answer_maps_ids_to_real_urls(retriever: SemanticRetriever) -> None:
    llm = FakeLLM()
    result = service(retriever, llm).answer("Which app has Flutterwave payments?")

    assert result.insufficient_evidence is False
    assert result.answer == "Irfan built goCare."
    assert [c.url for c in result.citations] == [f"{SITE}#work"]
    assert result.citations[0].section == "goCare"
    prompt = llm.requests[0].messages[-1].text
    assert '<evidence id="S1"' in prompt and "Visitor question:" in prompt


def test_no_evidence_abstains_without_calling_the_llm(retriever: SemanticRetriever) -> None:
    llm = FakeLLM()
    strict = AnswerService(retriever, llm, AnswerSettings(min_score=0.99))

    result = strict.answer("Has Irfan worked with Rust?")  # relevant, but no evidence

    assert result.insufficient_evidence is True
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.citations == []
    assert llm.requests == []  # no paid call


def test_invented_citation_ids_are_dropped_and_flagged(retriever: SemanticRetriever) -> None:
    llm = FakeLLM(
        GroundedAnswer(
            answer="Irfan built 40 apps.", cited_sources=["S99"], insufficient_evidence=False
        )
    )
    result = service(retriever, llm).answer("Flutterwave payments")
    assert result.citations == []
    assert result.insufficient_evidence is True  # claimed facts with no valid source


def test_model_reported_insufficiency_is_preserved(retriever: SemanticRetriever) -> None:
    llm = FakeLLM(
        GroundedAnswer(
            answer="The portfolio doesn't say.", cited_sources=[], insufficient_evidence=True
        )
    )
    result = service(retriever, llm).answer("Flutterwave payments pricing")
    assert result.insufficient_evidence is True and result.citations == []


def test_redaction_placeholders_never_reach_the_visitor(retriever: SemanticRetriever) -> None:
    llm = FakeLLM(
        GroundedAnswer(
            answer="Email him at [email redacted].",
            cited_sources=["S1"],
            insufficient_evidence=False,
        )
    )
    result = service(retriever, llm).answer("Flutterwave payments")
    assert "redacted" not in result.answer
    assert "the Contact section" in result.answer


def test_follow_up_uses_history_for_retrieval_and_context(retriever: SemanticRetriever) -> None:
    llm = FakeLLM()
    history = [Turn(question="Tell me about the matchmaking app", answer="Irfan worked on Tegal.")]

    service(retriever, llm).answer("Which technologies did it use?", history)

    messages = llm.requests[0].messages
    assert [m.role for m in messages] == [Role.USER, Role.MODEL, Role.USER]
    assert messages[0].text == "Tell me about the matchmaking app"
    assert "Tegal" in messages[-1].text  # follow-up retrieval found the referenced unit


def test_history_is_bounded(retriever: SemanticRetriever) -> None:
    llm = FakeLLM()
    history = [Turn(question=f"q{i} payments", answer="a" * 5000) for i in range(10)]
    service(retriever, llm, history_turns=2, history_answer_chars=100).answer("payments", history)
    messages = llm.requests[0].messages
    assert len(messages) == 5  # 2 turns x 2 + current
    assert len(messages[1].text) == 100


def test_llm_errors_propagate_typed(retriever: SemanticRetriever) -> None:
    with pytest.raises(LLMTimeoutError):
        service(retriever, FakeLLM(error=LLMTimeoutError("slow"))).answer("Flutterwave payments")


@pytest.mark.parametrize(
    "question",
    [
        "Who won the football World Cup?",
        "What is the weather today?",
        "Tell me a joke.",
        "What is 245 * 17?",
        "Write an essay about climate change",
        "What's the best pizza topping?",  # no general pattern, no portfolio signal or evidence
    ],
)
def test_irrelevant_questions_get_the_scope_message_without_llm(
    retriever: SemanticRetriever, question: str
) -> None:
    from portfolio_ai.rag.scope import OUT_OF_SCOPE_ANSWER

    llm = FakeLLM()
    result = service(retriever, llm).answer(question)

    assert result.out_of_scope is True
    assert result.answer == OUT_OF_SCOPE_ANSWER
    assert result.citations == []
    assert result.insufficient_evidence is True
    assert llm.requests == []  # no Gemini call


def test_relevant_question_without_evidence_keeps_insufficient_evidence(
    retriever: SemanticRetriever,
) -> None:
    llm = FakeLLM()
    result = AnswerService(retriever, llm, AnswerSettings(min_score=0.99)).answer(
        "What is his hourly rate?"
    )
    assert result.out_of_scope is False
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert llm.requests == []


def test_off_topic_wording_about_irfan_is_still_answered_from_evidence(
    retriever: SemanticRetriever,
) -> None:
    llm = FakeLLM()
    result = service(retriever, llm).answer("Has Irfan built apps with Flutterwave payments?")
    assert result.out_of_scope is False and len(llm.requests) == 1


def test_short_follow_up_in_a_conversation_is_in_scope(retriever: SemanticRetriever) -> None:
    llm = FakeLLM()
    history = [
        Turn(
            question="Tell me about the goCare telemedicine app with Agora video calls",
            answer="goCare is a telemedicine app.",
        )
    ]
    result = service(retriever, llm).answer("Which one?", history)
    assert result.out_of_scope is False
    assert len(llm.requests) == 1
