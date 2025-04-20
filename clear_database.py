from qdrant_client import QdrantClient
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Qdrant
from logger_utils import setup_logger

QDRANT_COLLECTION_NAME = "your_collection_name"
logger = setup_logger()

def main():
    return clear_database()

def clear_database():
    try:
        logger.info("✨ Clearing Database")

        # Connect to Qdrant server
        qdrant_client = QdrantClient(host="localhost", port=6333)

        # Delete the entire collection (equivalent to clearing all vectors)
        qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
        logger.info(f"🧹 Deleted collection: {QDRANT_COLLECTION_NAME}")

        # Optionally, recreate the collection if needed (so the app doesn't break later)
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={"size": 384, "distance": "Cosine"}  # adapt size if different
        )
        logger.info(f"📦 Recreated collection: {QDRANT_COLLECTION_NAME}")

        # Clear tracked files
        with open("utils/files.txt", "w") as file:
            file.write("")
        logger.info("📁 Cleared files.txt")

        return True

    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        return False

if __name__ == "__main__":
    main()
