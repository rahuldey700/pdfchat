from pdfchat.UploadEngine import PineconeEngine
from pdfchat.RetrievalEngine.RetrievalEngine import RetrievalEngine
import io
import asyncio
from llama_index.core.response.pprint_utils import pprint_source_node
from llama_index.llms.openai import OpenAI
import time
from pdfchat.Agents.AgentOpenAI import GetAgentOpenAI
from pdfchat.TextTransform import ExplanationTransformer
import pprint

_prompt = """Based on the given excerpts from a document, answer the given question:
-----------------------------------
{excerpts}
-----------------------------------
Question: {question}
Answer:"""

# Create a PineconeEngine object
engine = PineconeEngine(
    texttransformer=ExplanationTransformer(),
)

# read the pdf as a Bytesio object
pdf = open("02.pdf", "rb")
pdf = io.BytesIO(pdf.read())

start = time.time()

# upload the pdf to the engine
chunks = asyncio.run(engine.parse(pdf, filename="somethingrandom3.pdf"))

# transform the chunks
asyncio.run(engine.transform(texts=[chunk.text for chunk in chunks]))

# index the pdf
asyncio.run(engine.index(chunks=chunks, namespace="somethingrandom3.pdf"))

end = time.time()

print(f"Time taken to index: {end-start}")

vectore_store = engine.as_llama_index_vextor_store(namespace="somethingrandom3.pdf")

retriever = RetrievalEngine(vector_store=vectore_store).as_retriever()

index = RetrievalEngine(vector_store=vectore_store).as_index()

query_eng = index.as_query_engine()

llm = OpenAI(model="gpt-3.5-turbo")

chat_engine = index.as_chat_engine(chat_mode="openai", llm=llm, verbose=True)

agent = GetAgentOpenAI(engine=engine, namespace="somethingrandom3.pdf", top_k=5).get_openai_agent(
    llm=llm, 
    sys_prompt="""As an expert proposal writer, your job is to help the user understand the given proposal.
    Use the tools at your disposal to provide responses to the users questions.
    Answer to the best of your capabilities.
    
    DO NOT:
    Provide unhelpful responses.
    Provide inaccurate information.
    Provide information outside the scope of the proposal.""",
    verbose=True
)

# while True:
#     query = input("Enter your query: ")
#     res = retriever.retrieve(query)
#     texts = [r.text for r in res]
#     print(len(texts))
#     prompt = _prompt.format(excerpts="\n".join(texts), question=query)
#     res = llm.complete(prompt).text
#     print(res)

while True:
    query = input("Enter your query: ")
    res = agent.chat(query)
    print(res)