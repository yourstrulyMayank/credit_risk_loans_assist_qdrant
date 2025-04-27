from qdrant_client.models import Filter, FieldCondition, MatchValue
from logger_utils import setup_logger
from qdrant_client_init import get_qdrant_client 
from get_embedding_function import get_embedding_function
from langchain.schema import Document
logger = setup_logger()
client = get_qdrant_client()
embedding_function = get_embedding_function()
def search_relevant_chunks(question, filename, top_k=5):    
    
    question_embedding = embedding_function.embed_query(question)
    logger.info(f"Question_embedding: {question_embedding}")
    # 3. Search in Qdrant with filter on metadata.source
    search_result = client.search(
        collection_name="my_documents",  # <--- use your real collection name
        query_vector=question_embedding,
        limit=top_k,
        with_payload=True
        # filter=Filter(
        #     must=[
        #         FieldCondition(
        #             key="metadata.source",
        #             match=MatchValue(value=f"data\\new\\{filename}")
        #         )
        #     ]
        # )
    )
    logger.info(f"Search result: {search_result}")
    # 4. Collect matched payloads
    relevant_chunks = [
        Document(page_content=result.payload["page_content"], metadata=result.payload["metadata"])
        for result in search_result
    ]

    logger.info(f"Filename: {filename}")
    logger.info(f"Matched documents: {relevant_chunks}")
    
    return relevant_chunks
