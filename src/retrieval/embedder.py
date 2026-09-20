import os

from dotenv import load_dotenv
from openai import OpenAI

from src.config import EMBEDDING_MODEL

load_dotenv()


BASE_URL = "https://foundation-models.api.cloud.ru/v1"


class Embedder:
    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=os.environ["CLOUD_RU_API_KEY"],
            base_url=BASE_URL,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        return [item.embedding for item in response.data]
