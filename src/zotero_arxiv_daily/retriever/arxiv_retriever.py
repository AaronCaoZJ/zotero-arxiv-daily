from .base import BaseRetriever, register_retriever
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
import re
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180

# The RSS summary is prefixed with announcement metadata, e.g.
# "arXiv:2609.11977v1 Announce Type: new \nAbstract: <real abstract>".
# It has to go before the text reaches the reranker's embedding model.
ARXIV_ABSTRACT_PREFIX = re.compile(
    r"^arXiv:\S+\s+Announce Type:\s*\S+\s*Abstract:\s*", re.IGNORECASE
)


def _arxiv_id(paper_url: str) -> str:
    return paper_url.rstrip("/").rsplit("/", 1)[-1]


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[Any]:
        """Return today's announcement entries straight from the arXiv RSS feed.

        The feed already carries id, title, authors and abstract for every
        announcement, which covers everything reranking needs.  Fetching the
        same metadata again through the arXiv API used to cost one request per
        20 papers and was the source of the HTTP 429 failures, so it is gone.
        """
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        entries = [
            entry for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            entries = entries[:10]
        logger.info(f"Found {len(entries)} arxiv announcements")
        return entries

    def _to_candidate(self, entry: Any) -> Paper:
        """Build a Paper from one RSS entry, without downloading anything."""
        paper_id = entry.id.removeprefix("oai:arXiv.org:")
        # The feed packs every author into a single comma-separated string.
        authors = [
            name.strip()
            for author in entry.get("authors") or []
            for name in author.get("name", "").split(",")
            if name.strip()
        ]
        return Paper(
            source=self.name,
            title=entry.title,
            authors=authors,
            abstract=ARXIV_ABSTRACT_PREFIX.sub("", entry.get("summary", "")).strip(),
            url=f"https://arxiv.org/abs/{paper_id}",
            pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        )

    def retrieve_candidates(self) -> list[Paper]:
        return [self._to_candidate(entry) for entry in self._retrieve_raw_papers()]

    def hydrate(self, papers: list[Paper]) -> list[Paper]:
        """Download full text, for the reranked papers only."""
        for paper in tqdm(papers, desc="Fetching arxiv full text"):
            paper.full_text = _fetch_full_text(paper)
            sleep(1)
        return papers

    def convert_to_paper(self, raw_paper: Any) -> Paper:
        paper = self._to_candidate(raw_paper)
        paper.full_text = _fetch_full_text(paper)
        return paper


def _fetch_full_text(paper: Paper) -> str | None:
    full_text = extract_text_from_tar(paper)
    if full_text is None:
        full_text = extract_text_from_html(paper)
    if full_text is None:
        full_text = extract_text_from_pdf(paper)
    return full_text


def extract_text_from_html(paper: Paper) -> str | None:
    html_url = paper.url.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: Paper) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: Paper) -> str | None:
    source_url = f"https://arxiv.org/e-print/{_arxiv_id(paper.url)}"
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.url, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
