from abc import ABC, abstractmethod
from typing import List
from pdfchat.Chunker import Chunk


class Chunker(ABC):
    def __init__(
        self,
        filename
    ):
        self.filename = filename

    async def chunk(self, text, **kwargs):
        await self._chunk(text, **kwargs)

    @abstractmethod
    async def _chunk(self, text, **kwargs) -> List[Chunk]:
        pass