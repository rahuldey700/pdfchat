class BaseTextTransformer:
    def __init__(self):
        pass

    def _transform(self, text: str) -> str:
        """
        Transform the text into itself

        Args:
            text (str): The text to transform

        Returns:
            str: The transformed text
        """
        return text

    def transform(self, text: str) -> str:
        return self._transform(text)
    
    async def _async_transform(self, text: str) -> str:
        return text

    async def async_transform(self, text: str) -> str:
        return await self._async_transform(text)