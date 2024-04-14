
import os
from dotenv import load_dotenv
from pdfchat.Chunker import Chunk

load_dotenv()

import re
from io import BytesIO

from openai import OpenAI
from pinecone import Pinecone

from pdfchat.Chunker import SentenceChunker, PDFReaderUpdated
from pdfchat.UploadEngine import UploadEngine
from llama_index.vector_stores.pinecone import PineconeVectorStore
import uuid

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
        index_name: str = None,
        pinecone_api_key: str = None,
        openai_api_key: str = None,
        openai: OpenAI = None
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

    async def _index(
        self, 
        chunks: list[Chunk],
        namespace: str
    ) -> None:
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        metadatas = [
            {**metadata, "text": text}
            for text, metadata in zip(texts, metadatas)
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