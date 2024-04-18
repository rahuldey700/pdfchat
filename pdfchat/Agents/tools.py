from llama_index.agent.openai import OpenAIAgent
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from pdfchat.RetrievalEngine.RetrievalEngine import RetrievalEngine
from pdfchat.UploadEngine import PineconeEngine


class GetTools:
    def __init__(
        self,
        engine: PineconeEngine,
        namespace: str,
        top_k: int = 5,
    ):
        self.engine = engine
        self.namespace = namespace
        self.top_k = top_k
        self.vectore_store = engine.as_llama_index_vextor_store(namespace=namespace)
        self.retriever = self.vectore_store.as_retriever(similarity_top_k=top_k)
        self.retriever = RetrievalEngine(vector_store=self.vectore_store).as_retriever()
        self.index = RetrievalEngine(vector_store=self.vectore_store).as_index()
        self.index_query_eng = self.index.as_query_engine()

    def get_tools(self):
        self.individual_query_engine_tools = [QueryEngineTool(
                query_engine=self.index_query_eng,
                metadata=ToolMetadata(
                    name=f"proposal_document_index",
                    description=(
                        "useful for when you want to answer queries about the"
                        f" the proposal document"
                    ),
                ),
            ),
        ]

        self.query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=self.individual_query_engine_tools,
        )

        self.query_engine_tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=ToolMetadata(
                name="sub_question_query_engine",
                description=(
                    "useful for when you want to answer queries that require analyzing"
                    " multiple multiple questions that are related to each other."
                ),
            ),
        )

        self.tools = self.individual_query_engine_tools + [self.query_engine_tool]

        return self.tools
    
    def get_openai_agent(self, verbose: bool = True):
        return OpenAIAgent.from_tools(self.tools, verbose=verbose)