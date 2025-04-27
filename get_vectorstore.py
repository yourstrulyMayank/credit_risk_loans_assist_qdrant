# get_vectorstore.py
from qdrant_client import QdrantClient
# from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from get_embedding_function import get_embedding_function
from config import QDRANT_DB_PATH, COLLECTION_NAME
from qdrant_client.http.models import Distance, VectorParams
def get_qdrant_store():
    embedding_function = get_embedding_function()
    
    client = QdrantClient(
        url="http://localhost:6333",
        # path=QDRANT_DB_PATH,
        prefer_grpc=False,
    )

    if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),  # Adjust vector size if needed
        )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_function,
        retrieval_mode=RetrievalMode.DENSE,
    )
