import os
import shutil
import uuid
import filetype
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from load_images import process_images_to_pdf
from logger_utils import setup_logger
from get_vectorstore import get_qdrant_store
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import Filter, FieldCondition, MatchValue
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client_init import get_qdrant_client
logger = setup_logger()
qdrant_client = get_qdrant_client()
# Paths
DATA_PATH = "data"
NEW_DATA_PATH = os.path.join("data", "new")
AVAILABLE_FILES_PATH = os.path.join("utils", "files.txt")

def main():
    collection_name = "my_documents"
    ensure_collection_exists(qdrant_client, collection_name, vector_size=384)
    db = get_qdrant_store(collection_name=collection_name)
    populate_database(db)

def ensure_collection_exists(qdrant_client: QdrantClient, collection_name: str, vector_size: int = 384):
    collections = qdrant_client.get_collections().collections
    if collection_name not in [c.name for c in collections]:
        logger.info(f"Collection '{collection_name}' not found. Creating it...")
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"✅ Collection '{collection_name}' created.")
    else:
        logger.info(f"Collection '{collection_name}' already exists.")


def populate_database(db):
    
    # Load new documents
    logger.info("Loading new documents...")
    documents = load_documents()
    # ✅ Normalize metadata: make sure 'source' only contains the filename
    
    if not documents:
        logger.info("No new documents found.")
        print("No new documents found.")
        return
    for doc in documents:
        full_source = doc.metadata.get("source", "")
        doc.metadata["source"] = os.path.basename(full_source)
    new_sources = {doc.metadata['source'].split('\\')[-1] for doc in documents}
    logger.info(f"Found new sources: {new_sources}")

    print("Removing old entries for re-uploaded documents...")
    logger.info("Removing old entries for re-uploaded documents...")
    remove_existing_documents(db, new_sources)

    logger.info("Splitting documents into chunks...")
    chunks = split_documents(documents)

    logger.info("Adding chunks to Qdrant...")
    print("Adding to Database")
    add_to_qdrant(chunks, db)
    print("Added to Database")

    # Add to files.txt
    filename = documents[-1].metadata['source'].split('\\')[-1]
    add_file_to_list(db, filename, len(chunks))

    logger.info("Moving processed files to permanent storage...")
    for filename in os.listdir(NEW_DATA_PATH):
        src = os.path.join(NEW_DATA_PATH, filename)
        dest = os.path.join(DATA_PATH, filename)
        shutil.move(src, dest)
    print(f"All files moved from {NEW_DATA_PATH} to {DATA_PATH}")
    logger.info(f"All files moved from {NEW_DATA_PATH} to {DATA_PATH}")

def load_documents():
    document_loader = PyPDFDirectoryLoader(NEW_DATA_PATH)

    for filename in os.listdir(NEW_DATA_PATH):
        logger.info("Checking and processing files in NEW_DATA_PATH...")
        file_path = os.path.join(NEW_DATA_PATH, filename)

        if filename.endswith('.pdf'):
            if is_text_pdf(file_path):
                logger.info(f"Text-based PDF: {filename}")
            else:
                logger.info(f"OCR image-based PDF: {filename}")
                process_images_to_pdf(file_path)
                os.remove(file_path)
        elif is_image_file(file_path):
            logger.info(f"OCR image: {filename}")
            process_images_to_pdf(file_path)
            os.remove(file_path)

    logger.info("Loading all text-based PDFs in NEW_DATA_PATH.")
    return document_loader.load()

def is_text_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                if page.extract_text():
                    return True
        return False
    except Exception as e:
        logger.warning(f"PDF read error: {file_path} => {e}")
        return False

def is_image_file(file_path):
    kind = filetype.guess(file_path)
    return kind and kind.mime.startswith("image/")

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    logger.info(f"Splitting {len(documents)} documents in parallel...")
    def split_single(doc): return text_splitter.split_documents([doc])

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(split_single, documents))

    total_chunks = [chunk for sublist in results for chunk in sublist]
    logger.info(f"Total chunks created: {len(total_chunks)}")
    return total_chunks

def add_to_qdrant(chunks: list[Document], db, batch_size=50):
    

    chunks_with_ids = calculate_chunk_ids(chunks)
    scroll_result = db.client.scroll(
        collection_name=db.collection_name,
        with_payload=True,
        with_vectors=False,
        limit=10_000,
    )
    existing_ids = set([point.id for point in scroll_result[0]])

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if not new_chunks:
        logger.info("✅ No new documents to add")
        return

    logger.info(f"👉 Adding {len(new_chunks)} new documents to DB in batches")
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        batch_ids = [str(uuid.uuid4()) for _ in batch]
        # batch_ids = [chunk.metadata["id"] for chunk in batch]
        db.add_documents(batch, ids=batch_ids)
        logger.info(f"✅ Added batch {i // batch_size + 1} of {len(batch)} chunks")

    logger.info("✅ All new documents added")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        current_chunk_index = current_chunk_index + 1 if current_page_id == last_page_id else 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id

    return chunks

def add_file_to_list(db, file_name, new_chunk_count):
    print("Here in adding file to list")
    logger.info(f"Updating {AVAILABLE_FILES_PATH} with file: {file_name}")

    scroll_result = db.client.scroll(
        collection_name=db.collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=file_name))]
        ),
        with_payload=True,
        with_vectors=False,
        limit=10_000,
    )
    existing_chunks = scroll_result[0]  # This gives you the points directly

    existing_chunk_count = len(existing_chunks)
    chunk_difference = new_chunk_count - existing_chunk_count

    if os.path.exists(AVAILABLE_FILES_PATH):
        with open(AVAILABLE_FILES_PATH, "r") as file:
            lines = file.readlines()
        lines = [line for line in lines if not line.startswith(f"{file_name}:")]
    else:
        lines = []

    lines.append(f"{file_name}:{chunk_difference}\n")
    with open(AVAILABLE_FILES_PATH, "w") as file:
        file.writelines(lines)

    logger.info(f"✅ Updated file list for {file_name} with {chunk_difference} new chunks.")
    print(f"✅ Updated file list for {file_name} with {chunk_difference} new chunks.")

def remove_existing_documents(db, sources):
    client = db.client
    for source in sources:
        source_filename = source.split("\\")[-1]
        points = client.scroll(
            collection_name=db.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filename))]
            ),
            limit=10_000,
        )
        point_ids = [point.id for point in points[0]]
        if point_ids:
            logger.info(f"Deleting {len(point_ids)} points for {source_filename}...")
            client.delete(collection_name=db.collection_name, points=point_ids)
            logger.info(f"✅ Deleted entries for {source_filename}")

if __name__ == "__main__":
    main()
