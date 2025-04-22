# get_embedding_function.py
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME


def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
