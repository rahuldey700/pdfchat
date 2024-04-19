import os

from dotenv import load_dotenv

from pdfchat.Chunker import Chunk
from pdfchat.TextTransform import BaseTextTransformer

load_dotenv()

import asyncio
import re
import uuid
from io import BytesIO

from llama_index.vector_stores.pinecone import PineconeVectorStore
from openai import OpenAI
from pinecone import Pinecone

from pdfchat.Chunker import PDFReaderUpdated, SentenceChunker
from pdfchat.UploadEngine import UploadEngine


def standardize_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def get_embeddings(texts, openai, batch_size, model="text-embedding-ada-002"):
    texts = [standardize_text(text) for text in texts]
    embeddings = []
    for i in range(0, len(texts), batch_size):
        response = openai.embeddings.create(
            input=texts[i:i+batch_size],
            model=model
        )
        data = response.data
        for d in data:
            embeddings.append(d.embedding)
    return embeddings

class PineconeEngine(UploadEngine):
    BATCH_SIZE = 100

    def __init__(
        self,
        texttransformer: BaseTextTransformer = BaseTextTransformer(),
        index_name: str = None,
        pinecone_api_key: str = None,
        openai_api_key: str = None,
        openai: OpenAI = None,
    ):
        if not pinecone_api_key:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not index_name:
            self.index_name = os.getenv("PINECONE_INDEX_NAME")
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone = Pinecone(pinecone_api_key)
        self.pinecone_index = self.pinecone.Index(self.index_name)
        if not openai:
            openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.openai = openai
        self.transformed_texts = []
        self.texttransformer = texttransformer

    async def _upload(
        self,
        file: BytesIO
    ) -> None:
        pass

    async def _parse(
        self, 
        file: BytesIO,
        filename: str
    ) -> list[Chunk]:
        reader = PDFReaderUpdated()
        documents = reader.load_data_bytesio(file)
        return await SentenceChunker(
            filename=filename
        ).chunk(text=None, documents=documents)
    
    async def _transform(
        self,
        texts: list[str]
    ) -> list[str]:
        tasks = []
        for text in texts:
            tasks.append(
                asyncio.create_task(
                    self.texttransformer.async_transform(text)
                )
            )
        res = await asyncio.gather(*tasks)
        self.transformed_texts = res
        return res

    async def _index(
        self, 
        chunks: list[Chunk],
        namespace: str
    ) -> None:
        texts = self.transformed_texts
        chunk_texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        metadatas = [
            {**metadata, "text": text}
            for text, metadata in zip(chunk_texts, metadatas)
        ]
        vectors = get_embeddings(texts, self.openai, self.BATCH_SIZE)
        for i in range(0, len(vectors), self.BATCH_SIZE):
            self.pinecone_index.upsert(
                vectors=[
                    {
                        "id": str(uuid.uuid4()),
                        "values": vector,
                        "metadata": metadata
                    } for vector, metadata in zip(vectors[i:i+self.BATCH_SIZE], metadatas[i:i+self.BATCH_SIZE])
                ],
                namespace=namespace
            )

    async def _search(
        self, 
        query: str,
        namespace: str,
        top_k: int = 5,
        filter: dict = None
    ):
        vector = get_embeddings([query], self.openai)[0]
        if filter:
            return self.pinecone_index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=filter
            )
        return self.pinecone_index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
    
    def as_llama_index_vextor_store(self, namespace: str):
        return PineconeVectorStore(
            index_name=self.index_name,
            namespace=namespace,
            text_key="text"
        )

if __name__ == '__main__':
    pass