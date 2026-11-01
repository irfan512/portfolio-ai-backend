"""Guard the package boundary from context.md §5: `rag` must not depend on the API layer."""

import ast
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parents[2] / "src" / "portfolio_ai" / "rag"
FORBIDDEN_PREFIXES = ("fastapi", "starlette", "portfolio_ai.api", "portfolio_ai.main")


def imported_modules(path: Path) -> set[str]:
    """Return every module name a file imports (comments and docstrings are ignored)."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_rag_package_does_not_import_api_layer() -> None:
    offenders = [
        f"{path.relative_to(RAG_DIR)} imports {module}"
        for path in RAG_DIR.rglob("*.py")
        for module in imported_modules(path)
        if module.startswith(FORBIDDEN_PREFIXES)
    ]
    assert offenders == []
