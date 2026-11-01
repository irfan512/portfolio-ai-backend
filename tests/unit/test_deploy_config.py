"""Deployment config must match the image: catches drift like a PyTorch backend on an ONNX image."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def render_env() -> dict[str, dict]:
    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text())
    (service,) = blueprint["services"]
    return {var["key"]: var for var in service["envVars"]}


def dockerfile_env() -> dict[str, str]:
    return dict(
        re.findall(r"(PORTFOLIO_AI_[A-Z_]+|ORT_[A-Z_]+)=(\S+)", (ROOT / "Dockerfile").read_text())
    )


def test_render_embedding_backend_matches_image() -> None:
    image = dockerfile_env()
    assert image["PORTFOLIO_AI_EMBEDDING_BACKEND"] == "onnx"
    assert (
        render_env()["PORTFOLIO_AI_EMBEDDING_BACKEND"]["value"]
        == image["PORTFOLIO_AI_EMBEDDING_BACKEND"]
    )


def test_render_secrets_are_never_stored_in_the_blueprint() -> None:
    env = render_env()
    for key in ("GEMINI_API_KEY", "PORTFOLIO_AI_QDRANT_API_KEY", "PORTFOLIO_AI_QDRANT_URL"):
        assert env[key].get("sync") is False and "value" not in env[key]


def test_render_service_shape() -> None:
    (service,) = yaml.safe_load((ROOT / "render.yaml").read_text())["services"]
    assert (service["runtime"], service["region"], service["plan"]) == (
        "docker",
        "frankfurt",
        "starter",
    )
    assert service["healthCheckPath"] == "/ready"
    assert render_env()["PORTFOLIO_AI_CORS_ORIGINS"]["value"] == '["https://irfan512.github.io"]'


def test_image_disables_onnx_telemetry() -> None:
    assert dockerfile_env()["ORT_DISABLE_TELEMETRY"] == "1"
