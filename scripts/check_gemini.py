"""Verify Gemini configuration WITHOUT paid generation requests.

    python scripts/check_gemini.py [--env-file .env]

1. Checks GEMINI_API_KEY / GEMINI_MODEL are set (prints only their length).
2. `models.get`: the key is accepted and the model exists (metadata, no charge).
3. `count_tokens` on the exact prompts the live tests would send (no charge),
   so you can price `pytest -m gemini` before running it.
Needs the local stack for step 3 (docker compose up -d; portfolio-ai-ingest).
"""

import argparse
import sys
from pathlib import Path

from portfolio_ai.core.config import load_settings


class _CapturingLLM:
    """Records the prompts AnswerService would send, returns a canned answer."""

    model_name = "capture"

    def __init__(self) -> None:
        self.requests = []

    def generate_grounded(self, request):
        from portfolio_ai.rag.generation.base import GroundedAnswer

        self.requests.append(request)
        return GroundedAnswer(
            answer="(not sent)", cited_sources=["S1"], insufficient_evidence=False
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args()
    settings = load_settings(args.env_file)

    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    print(f"GEMINI_API_KEY  {'set (' + str(len(key)) + ' chars)' if key else 'MISSING'}")
    print(f"GEMINI_MODEL    {settings.gemini_model or 'MISSING'}")
    if not key or not settings.gemini_model:
        return 1

    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=15_000))
    try:
        model = client.models.get(model=settings.gemini_model)
    except errors.APIError as exc:
        print(f"models.get failed: HTTP {exc.code} ({exc.status}). Check the key and model id.")
        return 2
    print(
        f"model OK        {model.name}  input_limit={model.input_token_limit}  output_limit={model.output_token_limit}"
    )
    actions = list(getattr(model, "supported_actions", None) or [])
    print(f"supports        {', '.join(actions) or 'not reported'}")
    if actions and "generateContent" not in actions:
        print(
            "This model does not support generateContent, which the API uses. Choose another GEMINI_MODEL."
        )
        return 3

    # Prompts the live tests would send (tests/integration/test_gemini_live.py)
    from portfolio_ai.container import build_services
    from portfolio_ai.rag.service import AnswerService, Turn

    services = build_services(settings)
    if services.embedder is None or services.store is None:
        print("local stack not ready; skipping token estimate")
        return 0
    capture = _CapturingLLM()
    service = AnswerService(
        services.answer_service._retriever, capture, services.answer_service._settings
    )
    service.answer("Has Irfan built a telemedicine or healthcare app?")
    first = "Tell me about the goCare app."
    service.answer(first)
    service.answer(
        "Which payment provider did it use?", [Turn(question=first, answer="(previous answer)")]
    )
    service.answer("What is Irfan's hourly rate for a 3-month contract?")

    total = 0
    for i, request in enumerate(capture.requests, 1):
        contents = [
            types.Content(role=m.role.value, parts=[types.Part(text=m.text)])
            for m in request.messages
        ]
        try:
            counted = client.models.count_tokens(
                model=settings.gemini_model,
                contents=[
                    types.Content(role="user", parts=[types.Part(text=request.system_instruction)]),
                    *contents,
                ],
            )
        except errors.APIError as exc:
            print(
                f"count_tokens failed: HTTP {exc.code} ({exc.status}): {(exc.message or '')[:300]}"
            )
            return 4
        total += counted.total_tokens
        print(
            f"request {i}: ~{counted.total_tokens} input tokens (max output {request.max_output_tokens})"
        )
    print(
        f"live tests would make {len(capture.requests)} generate_content calls, ~{total} input tokens total"
    )
    print(
        f"worst case incl. retries: {len(capture.requests) * (settings.gemini_max_retries + 1)} calls"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
