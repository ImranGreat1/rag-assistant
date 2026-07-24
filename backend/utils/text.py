import os
from typing import List
from pypdf import PdfReader


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            raise ValueError("Provided path points to a directory and not a supported file type")
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_pdf(self.path)
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file(self.path)
        else:
            raise ValueError("Provided path is not a supported file (.pdf, .txt)")

    def load_pdf(self, path: str):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.documents.append(text)

    def load_file(self, path: str):
        with open(path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_document(self):
        self.load()
        return self.documents



class CharacterTextSplitter:
        def __init__(
            self, 
            chunk_size: int = 1000, 
            chunk_overlap: int = 200
        ):
            assert(
                chunk_size > chunk_overlap
            ), "Chunk size must be greater than chunk overlap"

            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split(self, text: str) -> List[str]:
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i : i + self.chunk_size])
            return chunks


        def split_texts(self, texts: List[str]) -> List[str]:
            chunks = []
            for text in texts:
                chunks.extend(self.split(text))
            return chunks

    




if __name__ == "__main__":
    document = TextFileLoader("data/HealthWellnessGuide.txt")
    text_splitter = CharacterTextSplitter()
    chunks = text_splitter.split_texts(document.load_document())

    print(len(chunks))
    print(chunks[0])
    print("-"*160)
    print(chunks[1])
    print("-"*160)
    print(chunks[-1])
