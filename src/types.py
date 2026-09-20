from typing import TypedDict


class Page(TypedDict):
    document_id: str
    page_idx: int
    page_number: int
    text: str


class Chunk(Page):
    chunk_id: str
    chunk_idx: int


class SearchResult(Chunk):
    score: float


class HybridSearchResult(SearchResult):
    rrf_score: float


class RerankedSearchResult(HybridSearchResult):
    rerank_score: float
