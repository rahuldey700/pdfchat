import io
from abc import ABC, abstractmethod
from pdfchat.Chunker import Chunk

class UploadEngine(ABC):
    def __init__(self) -> None:
        pass

    async def upload(
        self,
        file: io.BytesIO,
    ) -> None:
        await self._upload(file)

    async def parse(
        self,
        file: io.BytesIO,
    ) -> None:
        await self._parse(file)

    async def index(
        self,
        chunks: list[str],
    ) -> None:
        await self._index(chunks)

    async def search(
        self,
        query: str,
    ) -> None:
        await self._search(query)

    @abstractmethod
    async def _upload(self, file: io.BytesIO, **kwargs) -> None:
        pass

    @abstractmethod
    async def _parse(self, file: io.BytesIO, **kwargs) -> None:
        pass

    @abstractmethod
    async def _index(self, chunks: list[Chunk], namespace: str, **kwargs) -> None:
        pass

    @abstractmethod
    async def _search(self, query: str, namespace: str, **kwargs):
        pass