from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, RawPaperItem
from tqdm import tqdm
from typing import Type
from time import sleep
from loguru import logger


class BaseRetriever(ABC):
    name: str
    def __init__(self, config:DictConfig):
        self.config = config
        self.retriever_config = getattr(config.source,self.name)

    @abstractmethod
    def _retrieve_raw_papers(self) -> list[RawPaperItem]:
        pass

    @abstractmethod
    def convert_to_paper(self, raw_paper:RawPaperItem) -> Paper | None:
        pass

    def retrieve_papers(self) -> list[Paper]:
        raw_papers = self._retrieve_raw_papers()
        logger.info("Processing papers...")
        papers = []
        for raw_paper in tqdm(raw_papers, total=len(raw_papers), desc="Converting papers"):
            try:
                paper = self.convert_to_paper(raw_paper)
            except Exception as exc:
                logger.warning(f"Skipping paper {getattr(raw_paper, 'title', raw_paper)}: {exc}")
                continue
            if paper is not None:
                papers.append(paper)
            sleep(1)
        return papers

    def retrieve_candidates(self) -> list[Paper]:
        """Return rerank candidates without doing the expensive per-paper work.

        Reranking scores papers by ``abstract`` alone, so a source that can list
        abstracts cheaply should override this and defer downloads to
        :meth:`hydrate`.  The default keeps the old behaviour: do everything up
        front.
        """
        return self.retrieve_papers()

    def hydrate(self, papers: list[Paper]) -> list[Paper]:
        """Fill in the expensive fields (full text, ...) of the selected papers.

        Called only for the papers that survived reranking.  The default is a
        no-op, which is correct for retrievers whose ``retrieve_candidates``
        already returns fully populated papers.
        """
        return papers

registered_retrievers = {}

def register_retriever(name:str):
    def decorator(cls):
        registered_retrievers[name] = cls
        cls.name = name
        return cls
    return decorator

def get_retriever_cls(name:str) -> Type[BaseRetriever]:
    if name not in registered_retrievers:
        raise ValueError(f"Retriever {name} not found")
    return registered_retrievers[name]
