
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

def standardize_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def get_embeddings(texts, openai, batch_size, model="text-embedding-3-small"):
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
    def __init__(
        self,
        index_name: str = None,
        pinecone_api_key: str = None,
        openai_api_key: str = None,
    ):
        if not pinecone_api_key:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not index_name:
            index_name = os.getenv("PINECONE_INDEX_NAME")
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone = Pinecone(pinecone_api_key)
        self.index = self.pinecone.Index(index_name)
        if not openai:
            openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.openai = openai
        super().__init__()

    async def _upload(
        self,
        file: BytesIO
    ) -> None:
        pass

    async def _parse(
        self, 
        file: BytesIO
    ) -> None:
        reader = PDFReaderUpdated()
        documents = reader.load_data_bytesio(file)
        return await SentenceChunker.chunk(documents=documents, text=None)

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
        vectors = get_embeddings(texts, self.openai)
        for i in range(0, len(vectors), self.BATCH_SIZE):
            self.index.upsert(
                vectors=[
                    {
                        "id": str(i),
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
        vector = get_embeddings([query], self.llm)[0]
        if filter:
            return self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=filter
            )
        return self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
    
    def as_llama_index_vextor_store(self, index: str, namespace: str):
        return PineconeVectorStore(
            index_name=index, 
            namespace=namespace,
            text_key="text"
        )

if __name__ == '__main__':
    pass