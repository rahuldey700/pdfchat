from pdfchat.TextTransform import BaseTextTransformer
from llama_index.llms.openai import OpenAI

TRANSFORM_PROMPT = (
    "As an expert proposal writer, you are required to generate an explanation for the following text.\n"
    "The text is taken from a proposal document and is intended to act as instructional material for bidders.\n"
    "Generate an explanation describing the contents of the text in a way that is easy to understand.\n"
    "For example, you could explain the text using better phrases to add context and make it more understandable.\n"
    "Text: \n"
    "{text}\n"
    "Explanation:"
)

class ExplanationTransformer(BaseTextTransformer):

    def __init__(
        self,
        llm: OpenAI = OpenAI(model="gpt-3.5-turbo")
    ):
        super().__init__()
        self.llm = llm

    def _transform(self, text: str) -> str:
        prompt = TRANSFORM_PROMPT.format(text=text)
        response = self.llm.complete(prompt)
        return response.text
    
    async def _async_transform(self, text: str) -> str:
        prompt = TRANSFORM_PROMPT.format(text=text)
        response = await self.llm.acomplete(prompt)
        return response.text
    
    def transform(self, text: str) -> str:
        return self._transform(text)
    
    async def async_transform(self, text: str) -> str:
        return await self._async_transform(text)