from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from pdfchat.Chunker import Chunker, Chunk
from typing import List
import asyncio
from llama_index.readers.file import PDFReader

class SentenceChunker(Chunker):
    def __init__(
        self,
        filename: str = None,
        chunk_size: int = 256,
    ):
        self.sentence_splitter = SentenceSplitter(chunk_size=chunk_size)
        super().__init__(filename=filename)

    async def _chunk(
            self, 
            text: str = None,
            documents: List[Document] = None,
        ) -> List[Chunk]:
        """
        Chunk text into smaller chunks

        :param text: text
        :return: list of chunks
        """
        if text:
            documents = [Document(text=text)]
        nodes = self.sentence_splitter.get_nodes_from_documents(documents)
        return [
            Chunk(
                text=node.text,
                metadata={
                    'chunk_type': 'sentence',
                    'chunk_size': len(node.text),
                    'page_number': node.metadata['page_label'],
                    'chunk_number': i,
                    'filename': self.filename,
                }
            ) 
            for i, node in enumerate(nodes)
        ]

if __name__ == '__main__':
    sc = SentenceChunker(filename='test2.pdf')
    pdfr = PDFReader()
    documents = pdfr.load_data('test2.pdf')
    asyncio.run(sc.chunk(documents=documents, text=None))