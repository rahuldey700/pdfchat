from typing import List

import markdown
from bs4 import BeautifulSoup

from pdfchat.Chunker import Chunk, Chunker


class MarkdownChunker(Chunker):
    def __init__(self):
        super().__init__()

    async def _chunk(self, text: str, max_chunk_size: int = 10) -> List[Chunk]:
        """
        Chunk markdown text into smaller chunks

        :param text: markdown text
        :param max_chunk_size: maximum chunk size
        :return: list of chunks
        """
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, 'html.parser')
        chunks = []
        next_chunk = []
        # go through each element in the html
        for element in soup.find_all():

            if len(next_chunk) >= max_chunk_size:
                chunks.append("\n".join(next_chunk))
                next_chunk = []

            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if len(next_chunk) > 0:
                    chunks.append("\n".join(next_chunk))
                    next_chunk = []
                next_chunk.append(element.text)
            else:
                if len(next_chunk) > 0:
                    next_chunk.append(element.text)
                else:
                    chunks.append(element.text)
        if len(next_chunk) > 0:
            chunks.append("\n".join(next_chunk))
        return [
            Chunk(
                text=chunk,
                metadata={
                    'chunk_type': 'markdown',
                    'chunk_size': len(chunk),
                    'page_number': None,
                    'chunk_number': i,
                    'filename': self.filename,
                }
            ) 
            for i, chunk in enumerate(chunks)
        ]