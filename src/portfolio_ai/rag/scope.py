"""Is a question about Irfan's professional portfolio at all? Deterministic, no LLM.

PortfolioAI is not a general-purpose chatbot. Off-topic questions (sports results,
weather, jokes, maths, essays...) get a fixed, polite scope message, and they never
reach Gemini.

Precedence, safest first:

1. The question names Irfan or the portfolio ("he", "his", "Irfan", "portfolio")
   -> IN SCOPE. Missing facts are then handled by the normal insufficient-evidence path.
2. It matches a clearly general request (joke, weather, "who won", arithmetic...)
   -> OUT OF SCOPE, before any retrieval.
3. It uses professional / technical vocabulary (projects, Flutter, AI, payments,
   healthcare...) or is a follow-up in an ongoing conversation -> IN SCOPE.
4. Retrieval found evidence above the score threshold that shares a meaningful
   word with the question (e.g. a project name) -> IN SCOPE.
5. Otherwise -> OUT OF SCOPE.

A low retrieval score alone never rejects a question: rule 5 only applies when
none of the scope signals in rules 1, 3 and 4 is present.
"""

import re
from collections.abc import Iterable, Sequence
from enum import StrEnum

from portfolio_ai.rag.retrieval.base import SearchResult

OUT_OF_SCOPE_ANSWER = (
    "Thanks for your question! I'm here to help you explore Irfan's professional portfolio, "
    "including his projects, technical skills, experience, and services. Your question is "
    "outside that scope, but I'd be happy to help with anything related to his work."
)


class Scope(StrEnum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"


_WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")

# Rule 1: the question is about the portfolio owner or the portfolio itself.
_PERSON = re.compile(r"\b(irfan|haider|irfan's|he|him|his|he's|portfolio|this site|your work)\b")

# Rule 2: clearly general requests. Kept specific on purpose.
_OFF_TOPIC = re.compile(
    r"|".join(
        [
            r"\bjokes?\b",
            r"\b(poem|poetry|haiku|limerick|riddle|lyrics)\b",
            r"\bwrite (me )?(an? )?(essay|story|poem|song|letter|speech)\b",
            r"\bessay (on|about)\b",
            r"\b(weather|forecast|temperature|rain(ing)?|snow(ing)?)\b",
            r"\bworld cup\b",
            r"\b(football|soccer|cricket|nba|nfl|fifa|olympics?|premier league)\b",
            r"\bwho won\b",
            r"\b(election|president|prime minister)\b",
            r"\bcapital of\b",
            r"\b(recipe|cook|bake)\b",
            r"\b(horoscope|zodiac|lottery)\b",
            r"\bwhat time is it\b",
            r"\bwhat('?s| is) (the )?(date|day) today\b",
            r"\b(movie|film|song|tv show)s? (to watch|recommendations?)\b",
            r"\btranslate\b",
            r"\b(solve|calculate|integrate|differentiate|derivative|integral|equation)\b",
            r"\d+\s*(\+|-|\*|/|x|×|÷|\^)\s*\d+",
            r"\bmeaning of life\b",
        ]
    )
)

# Rule 3: professional and technical vocabulary that signals the question is about work.
_DOMAIN_TERMS = frozenset(
    """
    project projects work worked working experience experienced skill skills skillset
    service services offer offers career job jobs role roles position company companies employer
    employers education degree university study studied qualification qualifications background
    hire hiring freelance freelancer contract consultant consulting available availability rate rates
    pricing price cost quote contact email resume cv cv's linkedin github client clients
    app apps application applications mobile web website websites frontend backend fullstack
    full-stack admin dashboard dashboards platform platforms product products build built develop
    developed developer developers development engineer engineering software programming code
    coding tech technology technologies stack stacks framework frameworks tool tools language languages
    flutter dart react javascript typescript python kotlin java swift swiftui ios android
    firebase laravel php mysql graphql rest api apis websocket websockets real-time realtime
    ai ml llm llms machine learning model models nlp speech chatbot chatbots agent agents rag
    tensorflow tflite automation integration integrations
    payment payments stripe flutterwave agora video voice chat streaming notifications
    healthcare telemedicine health medical e-sports esports gaming social networking matchmaking
    real estate fintech education edtech
    ci cd devops deployment testing release
    """.split()
)
_DOMAIN_PHRASES = ("machine learning", "real estate", "app store", "google play", "play store")

# Follow-ups that only make sense with the previous turn.
_FOLLOW_UP = re.compile(
    r"\b(it|its|that|those|these|this|they|them|their|which one|which of|what about|how about|"
    r"and the|also|more|else|another|other)\b"
)

_STOPWORDS = frozenset(
    """
    a an the and or but if then so to of in on at by for with from about into over under is are was
    were be been being do does did done have has had can could should would will shall may might must
    what which who whom whose when where why how i me my we our you your they them their it its this
    that these those there here not no yes any some all each every more most much many very just
    tell show give list explain describe please thanks thank hello hi hey know think
    """.split()
)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _content_words(text: str) -> set[str]:
    return {w.strip(".-") for w in _words(text) if len(w) >= 3 and w not in _STOPWORDS}


def mentions_portfolio_owner(question: str) -> bool:
    return bool(_PERSON.search(question.lower()))


def is_clearly_off_topic(question: str) -> bool:
    return bool(_OFF_TOPIC.search(question.lower()))


def has_domain_vocabulary(question: str) -> bool:
    lowered = question.lower()
    if any(phrase in lowered for phrase in _DOMAIN_PHRASES):
        return True
    return any(word.strip(".-") in _DOMAIN_TERMS for word in _words(lowered))


def is_follow_up(question: str) -> bool:
    return len(_words(question)) <= 8 or bool(_FOLLOW_UP.search(question.lower()))


def evidence_overlaps(question: str, results: Iterable[SearchResult], top_n: int = 3) -> bool:
    """True when retrieved evidence shares a meaningful word with the question."""
    asked = _content_words(question)
    if not asked:
        return False
    for result in list(results)[:top_n]:
        if asked & _content_words(result.chunk.text):
            return True
    return False


def classify_before_retrieval(question: str) -> Scope | None:
    """Rules 1-2: decide early when possible (no embedding, no Qdrant call). None = undecided."""
    if mentions_portfolio_owner(question):
        return Scope.IN_SCOPE
    if is_clearly_off_topic(question):
        return Scope.OUT_OF_SCOPE
    return None


def classify(question: str, *, has_history: bool, results: Sequence[SearchResult]) -> Scope:
    """Rules 1-5. `results` are the retrieval results already above the score threshold."""
    early = classify_before_retrieval(question)
    if early is not None:
        return early
    if has_domain_vocabulary(question):
        return Scope.IN_SCOPE
    if has_history and is_follow_up(question):
        return Scope.IN_SCOPE
    if evidence_overlaps(question, results):
        return Scope.IN_SCOPE
    return Scope.OUT_OF_SCOPE
