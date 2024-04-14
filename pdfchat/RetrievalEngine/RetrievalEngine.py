
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore


# inherits from a llama_index class
class RetrievalEngine:
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        top_k: int = 5,
    ):
        assert vector_store.namespace != None, "Namespace must be set in the vector store"
        self.vector_store = vector_store
        self.retriever = VectorStoreIndex.from_vector_store(self.vector_store).as_retriever(
            similarity_top_k=top_k
        )

    def as_retriever(self):
        return self.retriever