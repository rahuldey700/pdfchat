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