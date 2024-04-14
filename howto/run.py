from pdfchat.UploadEngine import PineconeEngine
from pdfchat.RetrievalEngine.RetrievalEngine import RetrievalEngine
import io
import asyncio
from llama_index.core.response.pprint_utils import pprint_source_node
from llama_index.llms.openai import OpenAI
import time

_prompt = """Based on the given excerpts from a document, answer the given question:
-----------------------------------
{excerpts}
-----------------------------------
Question: {question}
Answer:"""

# Create a PineconeEngine object
engine = PineconeEngine()

# read the pdf as a Bytesio object
pdf = open("test02.pdf", "rb")
pdf = io.BytesIO(pdf.read())

start = time.time()

# upload the pdf to the engine
chunks = asyncio.run(engine.parse(pdf, filename="test0222.pdf"))

# index the pdf
asyncio.run(engine.index(chunks=chunks, namespace="test0222.pdf"))

end = time.time()

print(f"Time taken to index: {end-start}")

vectore_store = engine.as_llama_index_vextor_store(namespace="test0222.pdf")

retriever = RetrievalEngine(vector_store=vectore_store).as_retriever()

index = RetrievalEngine(vector_store=vectore_store).as_index()

query_eng = index.as_query_engine()

llm = OpenAI(model="gpt-3.5-turbo")

chat_engine = index.as_chat_engine(chat_mode="openai", llm=llm, verbose=True)

while True:
    query = input("Enter your query: ")
    res = retriever.retrieve(query)
    texts = [r.text for r in res]
    print(len(texts))
    prompt = _prompt.format(excerpts="\n".join(texts), question=query)
    res = llm.complete(prompt).text
    print(res)