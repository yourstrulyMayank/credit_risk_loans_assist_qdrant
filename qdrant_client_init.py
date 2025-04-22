from qdrant_client import QdrantClient
from config import QDRANT_DB_PATH

def get_qdrant_client():
    return QdrantClient(host="localhost", port=6333)
    # return QdrantClient(path=QDRANT_DB_PATH)