"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def _new_entries(feed):
    return [e for e in feed.entries if e.get("arxiv_announce_type", "new") == "new"]


def _explode(*args, **kwargs):
    raise AssertionError("listing candidates must not hit the network")


def test_retrieve_candidates_needs_no_downloads(config, mock_feedparser, monkeypatch):
    """Candidates come from the RSS feed alone: no arXiv API, no file downloads."""
    monkeypatch.setattr(arxiv_retriever, "_download_file", _explode)
    monkeypatch.setattr(arxiv_retriever, "_fetch_full_text", _explode)

    candidates = ArxivRetriever(config).retrieve_candidates()

    expected = _new_entries(mock_feedparser)
    assert len(candidates) == len(expected)
    assert {c.title for c in candidates} == {e.title for e in expected}
    assert all(c.source == "arxiv" for c in candidates)
    assert all(c.full_text is None for c in candidates)
    assert all(c.authors for c in candidates)


def test_retrieve_candidates_strips_announce_prefix(config, mock_feedparser):
    """The RSS boilerplate must not reach the reranker's embedding model."""
    candidates = ArxivRetriever(config).retrieve_candidates()

    assert candidates, "fixture should yield at least one 'new' entry"
    for candidate in candidates:
        assert candidate.abstract
        assert not candidate.abstract.startswith("arXiv:")
        assert "Announce Type:" not in candidate.abstract


def test_retrieve_candidates_derives_urls_from_id(config, mock_feedparser):
    candidates = ArxivRetriever(config).retrieve_candidates()
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in _new_entries(mock_feedparser)]

    assert [c.url for c in candidates] == [f"https://arxiv.org/abs/{i}" for i in paper_ids]
    assert [c.pdf_url for c in candidates] == [f"https://arxiv.org/pdf/{i}" for i in paper_ids]


def test_retrieve_candidates_can_include_cross_list(config, mock_feedparser):
    config.source.arxiv.include_cross_list = True

    candidates = ArxivRetriever(config).retrieve_candidates()

    assert len(candidates) == len(mock_feedparser.entries)


def test_hydrate_fetches_full_text_only_for_the_papers_given(config, mock_feedparser, monkeypatch):
    """The whole point of the split: downloads happen after reranking, not before."""
    config.source.arxiv.include_cross_list = True
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    fetched: list[str] = []

    def _fake_fetch(paper):
        fetched.append(paper.url)
        return f"full text of {paper.title}"

    monkeypatch.setattr(arxiv_retriever, "_fetch_full_text", _fake_fetch)

    retriever = ArxivRetriever(config)
    candidates = retriever.retrieve_candidates()
    assert len(candidates) > 2, "fixture should provide papers to leave behind"
    selected = candidates[:2]

    retriever.hydrate(selected)

    assert fetched == [p.url for p in selected]
    assert all(p.full_text is not None for p in selected)
    assert all(p.full_text is None for p in candidates[2:])


def test_retrieve_papers_still_populates_full_text(config, mock_feedparser, monkeypatch):
    """The one-shot path stays intact for callers that don't split the work."""
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: "pdf text")

    papers = ArxivRetriever(config).retrieve_papers()

    assert len(papers) == len(_new_entries(mock_feedparser))
    assert all(p.full_text == "pdf text" for p in papers)


def test_extract_text_from_tar_uses_eprint_url(config, monkeypatch):
    """Tar extraction derives the e-print URL from the abs URL."""
    captured: dict = {}

    def _fake_timeout(func, args, *, timeout, operation, paper_title):
        captured["args"] = args
        return "tex"

    monkeypatch.setattr(arxiv_retriever, "_run_with_hard_timeout", _fake_timeout)
    paper = SimpleNamespace(url="https://arxiv.org/abs/2508.13426v1", title="T", pdf_url=None)

    assert arxiv_retriever.extract_text_from_tar(paper) == "tex"
    assert captured["args"][0] == "https://arxiv.org/e-print/2508.13426v1"


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
