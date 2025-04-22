# get_vectorstore.py
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from get_embedding_function import get_embedding_function
from config import QDRANT_DB_PATH, COLLECTION_NAME

def get_qdrant_store():
    embedding_function = get_embedding_function()
    
    client = QdrantClient(
        url="http://localhost:6333",
        # path=QDRANT_DB_PATH,
        prefer_grpc=False,
    )

    if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": 384, "distance": "Cosine"}  # Adjust vector size if needed
        )

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_function,
    )
