# query_pipeline.py
from get_vectorstore import get_qdrant_store
from langchain_ollama import OllamaLLM
from config import LLM_MODEL_NAME
from logger_utils import setup_logger
# llm = OllamaLLM(model=LLM_MODEL_NAME)
logger = setup_logger()
def hybrid_query(question: str, filename: str, llm: OllamaLLM, top_k=5) -> str:
    db = get_qdrant_store()
    matched_docs = db.similarity_search(
    query=question,
    k=top_k,
    filter={
        "must": [
            {
                "key": "source",
                "match": {
                    "value": filename
                }
            }
        ]
    }
)
    logger.info(f'filename: {filename}')
    logger.info(f"Matched documents: {matched_docs}")
    relevant_chunks = [
        doc for doc in matched_docs if filename in doc.metadata.get("source", "")
    ]

    if not relevant_chunks:
        return "Not Specified in the Document"

    context = "\n---\n".join([doc.page_content for doc in relevant_chunks])
    prompt = f"{question}\n\nContext:\n{context}\n\nGive a concise answer in a short phrase:"

    try:
        return llm.invoke(prompt).strip()
    except Exception as e:
        return f"[Error] {str(e)}"
