import uuid

from infra.pinecone import PineconeDB
from utils.text import CharacterTextSplitter, TextFileLoader

if __name__ == "__main__":
    document = TextFileLoader("data/HealthWellnessGuide.txt")
    text_splitter = CharacterTextSplitter()
    chunks = text_splitter.split_texts(document.load_document())
    formatted_chunks = [{"_id": str(uuid.uuid4()), "content": chunk} for chunk in chunks]

    pc = PineconeDB()
    index = pc.create_or_get_index(name="rag-assistance")
    index.upsert_records(namespace="health-wellness-guide", records=formatted_chunks)
