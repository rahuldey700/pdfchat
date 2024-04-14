from io import BytesIO
from typing import Dict, List

from fsspec import AbstractFileSystem
from llama_index.core import Document
from llama_index.readers.file import PDFReader


class PDFReaderUpdated(PDFReader):
    def __init__(self, return_full_document: bool | None = False) -> None:
        super().__init__(return_full_document)

    def load_data_bytesio(
        self,
        file: BytesIO,
        extra_info: Dict | None = None,
        fs: AbstractFileSystem | None = None,
    ) -> List[Document]:
        """Parse file."""
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required to read PDF files: `pip install pypdf`"
            )
        stream = file
        pdf = pypdf.PdfReader(stream)
        num_pages = len(pdf.pages)
        docs = []
        if self.return_full_document:
            text = ""
            metadata = {"file_name": file.name}
            for page in range(num_pages):
                page_text = pdf.pages[page].extract_text()
                text += page_text
            docs.append(Document(text=text, metadata=metadata))
        else:
            for page in range(num_pages):
                page_text = pdf.pages[page].extract_text()
                page_label = pdf.page_labels[page]
                metadata = {"page_label": page_label, "file_name": file.name}
                if extra_info is not None:
                    metadata.update(extra_info)
                docs.append(Document(text=page_text, metadata=metadata))
        return docs