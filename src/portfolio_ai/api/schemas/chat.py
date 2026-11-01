"""Public chat API contract (see docs/api.md)."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SessionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{16,64}$")]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    session_id: SessionId | None = None


class CitationOut(BaseModel):
    title: str = Field(examples=["Selected work"])
    section: str = Field(examples=["goCare"])
    url: str = Field(examples=["https://irfan512.github.io/portfolio/#work"])


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    session_id: SessionId
    insufficient_evidence: bool
    out_of_scope: bool = False  # question is not about the portfolio (citations are empty)
