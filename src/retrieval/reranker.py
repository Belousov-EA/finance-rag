import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from src.config import BASE_URL, RERANKER_MODEL

load_dotenv()


class Reranker:
    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=os.environ["CLOUD_RU_API_KEY"],
            base_url=BASE_URL,
        )

    def rerank(
        self,
        query: str,
        results: list[dict],
        limit: int | None = None,
    ) -> list[dict]:
        if not results:
            return []

        documents = [result["text"] for result in results]

        response = self.client.post(
            path="/score",
            cast_to=httpx.Response,
            body={
                "model": RERANKER_MODEL,
                "encoding_format": "float",
                "text_1": query,
                "text_2": documents,
            },
        )

        scores = response.json()["data"]

        reranked = []

        for item in scores:
            idx = item["index"]

            reranked.append(
                {
                    **results[idx],
                    "rerank_score": item["score"],
                }
            )

        reranked.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        if limit is not None:
            reranked = reranked[:limit]

        return reranked
