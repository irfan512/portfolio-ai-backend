from portfolio_ai.rag.domain.documents import SourceType, TextBlock
from portfolio_ai.rag.domain.ids import content_hash_for, document_id_for

URI = "https://irfan512.github.io/portfolio/#work"


def test_document_id_is_stable_and_depends_on_identity() -> None:
    assert document_id_for(SourceType.WEBSITE, URI) == document_id_for(SourceType.WEBSITE, URI)
    assert document_id_for(SourceType.WEBSITE, URI) != document_id_for(
        SourceType.WEBSITE, URI.replace("#work", "#about")
    )
    assert document_id_for(SourceType.WEBSITE, URI) != document_id_for(SourceType.MARKDOWN, URI)


def test_content_hash_covers_text_and_headings() -> None:
    base = [TextBlock(text="Flutter", heading_path=("Alpha",))]
    assert content_hash_for(base) == content_hash_for(list(base))
    assert content_hash_for(base) != content_hash_for(
        [TextBlock(text="Flutter!", heading_path=("Alpha",))]
    )
    assert content_hash_for(base) != content_hash_for(
        [TextBlock(text="Flutter", heading_path=("Beta",))]
    )
