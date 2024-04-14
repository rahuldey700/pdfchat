import io
from abc import ABC, abstractmethod
from pdfchat.Chunker import Chunk

class UploadEngine(ABC):
    def __init__(self) -> None:
        pass

    async def upload(
        self,
        file: io.BytesIO,
        **kwargs,
    ) -> None:
        return await self._upload(file, **kwargs)

    async def parse(
        self,
        file: io.BytesIO,
        **kwargs,
    ) -> None:
        return await self._parse(file, **kwargs)

    async def index(
        self,
        chunks: list[Chunk],
        namespace: str,
        **kwargs,
    ) -> None:
        return await self._index(chunks, namespace, **kwargs)

    async def search(
        self,
        query: str,
        namespace: str,
        **kwargs,
    ) -> None:
        return await self._search(query, namespace, **kwargs)

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