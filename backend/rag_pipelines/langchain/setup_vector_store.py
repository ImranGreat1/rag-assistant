import uuid

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

if __name__ == "__main__":
    load_dotenv()

    # Load document
    loader = TextLoader("data/HealthWellnessGuide.txt", encoding="utf-8")
    loaded_document = loader.load()

    # Split loaded document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(loaded_document[0].page_content)

    # Initialize local Qdrant client and store data on disk
    client = QdrantClient(path="./vector_store") # Local qdrant store on disk
    # client = QdrantClient(
    #     url=os.getenv("QDRANT_CLUSTER_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
    # )  # For cloud hosted Qdrant

    if client.collection_exists("rag-assistant"):
        client.delete_collection("rag-assistant")
    client.create_collection(
        collection_name="rag-assistant",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # Create documents for embedding and add it vector store
    documents = [Document(page_content=chunk) for chunk in chunks]
    uuids = [str(uuid.uuid4()) for _ in range(len(documents))]

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="rag-assistant",
        embedding=FastEmbedEmbeddings(model="BAAI/bge-small-en-v1.5"),
    )

    vector_store.add_documents(documents=documents, ids=uuids)
