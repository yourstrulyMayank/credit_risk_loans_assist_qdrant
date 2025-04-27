# query_pipeline.py
from get_vectorstore import get_qdrant_store
from langchain_ollama import OllamaLLM
from config import LLM_MODEL_NAME
from logger_utils import setup_logger
from qdrant_client.models import MatchText, Filter, FieldCondition, MatchValue
# llm = OllamaLLM(model=LLM_MODEL_NAME)
from qdrant_client_init import get_qdrant_client

from search_pipeline import search_relevant_chunks
logger = setup_logger()
qdrant_client = get_qdrant_client() 
db = get_qdrant_store()
def hybrid_query(question: str, filename: str, llm: OllamaLLM, top_k=5) -> str:
    
    logger.info(f'Question: {question}')
    source = "data\\new\\" + filename
    logger.info(f'Source: {source}')
    matched_docs = db.similarity_search(
        query=question,
        k=top_k,
        filter=Filter(
            should=[
                FieldCondition(
                    key="metadata.source", 
                    match=MatchValue(value=source)
                )
            ]
        )
    )
    logger.info(f'filename: {filename}')
    logger.info(f"Matched documents: {matched_docs}")
    relevant_chunks = [
        doc for doc in matched_docs if filename in doc.metadata.get("source", "")
    ]
    # relevant_chunks = search_relevant_chunks(question, filename, top_k=top_k)
    logger.info(f"Relevant chunks: {relevant_chunks}")
    if not relevant_chunks:
        return "Not Specified in the Document"

    context = "\n---\n".join([doc.page_content for doc in relevant_chunks])
    prompt = f"{question}\n\nContext:\n{context}\n\nGive a concise answer in a short phrase:"

    try:
        return llm.invoke(prompt).strip()
    except Exception as e:
        return f"[Error] {str(e)}"
