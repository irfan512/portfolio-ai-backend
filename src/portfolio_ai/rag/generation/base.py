"""Provider-neutral LLM contract. Gemini is one implementation; tests use a fake."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    USER = "user"
    MODEL = "model"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    text: str


class GroundedAnswer(BaseModel):
    """The JSON shape the model must return (enforced via structured output)."""

    answer: str = Field(description="Answer for the visitor, plain text.")
    cited_sources: list[str] = Field(
        default_factory=list, description="Ids of the evidence blocks used, e.g. ['S1', 'S3']."
    )
    insufficient_evidence: bool = Field(
        description="True when the evidence does not answer the question."
    )


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    system_instruction: str
    messages: Sequence[Message]
    temperature: float | None = None  # None: model default (Gemini 3.x deprecates sampling params)
    max_output_tokens: int = 800


class LLMProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate_grounded(self, request: GenerationRequest) -> GroundedAnswer:
        """Return a validated `GroundedAnswer` or raise an `LLMError` subclass."""
        ...
