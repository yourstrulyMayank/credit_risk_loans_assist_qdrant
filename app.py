import os
import sys
import time
import subprocess
import threading
from flask import Flask, render_template, request, redirect, url_for, jsonify
from langchain_ollama import OllamaLLM
from langchain_qdrant import Qdrant
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
import populate_database
import clear_database
from get_embedding_function import get_embedding_function
from query_data import query_rag, query_rag_latest
from logger_utils import setup_logger
import asyncio
# from async_summary_pipeline import generate_summary_with_graph
from summary_utils import generate_summary
from query_pipeline import hybrid_query
from qdrant_client_init import get_qdrant_client
from populate_database import ensure_collection_exists
from get_vectorstore import get_qdrant_store
# ------------------- Config -------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'data/new'
PROMPTS_FILE_PATH = "utils/prompts.txt"
FILES_TRACK_PATH = "utils/files.txt"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
logger = setup_logger()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- Globals -------------------
embedding_function = get_embedding_function()
def start_qdrant():
    qdrant_path = "qdrant.exe"  # adjust path as needed
    subprocess.Popen([qdrant_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

start_qdrant()
time.sleep(2)
qdrant_client = get_qdrant_client()  # same as in populate_database
COLLECTION_NAME = "my_documents"

db = get_qdrant_store()
model = OllamaLLM(model="llama3.2")

processing_status_upload = {"complete": False}
processing_status_fetch = {"complete": False}
fetched_results = {}
latest_file_data = {}

# ------------------- Routes -------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file and file.filename:
        logger.info(f"Received file upload: {file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        logger.info(f"Saved file to: {filepath}")
        update_file_registry(file.filename)

        threading.Thread(target=run_populate_database, args=(file.filename,)).start()
        logger.info(f"Started background thread for: {file.filename}")
        return render_template('loading.html')
    return redirect(url_for('index'))


@app.route('/ask', methods=['GET', 'POST'])
def ask():
    document_titles = load_file_titles()
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            logger.info(f"Received question: {question}")
            response = query_rag(question, db, model)
            return render_template('ask.html', response=response, document_titles=document_titles)
    return render_template('ask.html', document_titles=document_titles)


@app.route('/batch_ask', methods=['POST'])
def batch_ask():
    questions = request.json.get('questions', [])
    logger.info(f"Received batch questions: {questions}")
    answers = [query_rag(q, db, model) for q in questions]
    return jsonify({"answers": answers})


@app.route('/clear_database', methods=['GET', 'POST'])
def clear_database_route():
    try:
        logger.info("Request to clear database received.")
        removed_files = clear_database.clear_database(db)
        sync_file_registry(removed_files)
        logger.info(f"Removed files: {removed_files}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/fetching_results', methods=['GET'])
def fetching_results():
    return render_template('fetching_results.html')


@app.route('/analyze', methods=['GET'])
def analyze():
    return render_template('analyze.html', data=fetched_results)


@app.route('/check_status_upload')
def check_status_upload():
    return jsonify({"complete": processing_status_upload["complete"]})


@app.route('/check_status_fetch')
def check_status_fetch():
    return jsonify({"complete": processing_status_fetch["complete"]})


# ------------------- Background Tasks -------------------
def run_populate_database(latest_filename):
    global processing_status_upload
    processing_status_upload["complete"] = False
    logger.info("Starting populate_database task...")

    try:
        # Ensure collection exists before doing anything
        ensure_collection_exists(qdrant_client, COLLECTION_NAME, vector_size=384)

        populate_database.populate_database(db)
        logger.info("Database population complete.")
    finally:
        processing_status_upload["complete"] = True
        threading.Thread(target=run_query_database, args=(latest_filename,)).start()



def run_query_database(latest_file):
    global fetched_results, processing_status_fetch
    processing_status_fetch["complete"] = False
    logger.info("Starting RAG query and summary generation...")

    try:
        prompts = load_prompts(PROMPTS_FILE_PATH)
        logger.info(f'prompts: {prompts}')
        results = {}

        for k, v in prompts.items():
            answer = hybrid_query(v, latest_file, model)
            logger.info(f'answer: {answer}')
            results[k] = answer
        logger.info(f'results: {results}')

        # You can still generate summary here too
        from summary_utils import generate_summary
        summary = generate_summary([], latest_file)  # Pass empty if using pre-chunked files
        results["Summary"] = summary

        fetched_results.update(results)
        processing_status_fetch["complete"] = True

    except Exception as e:
        logger.error(f"Error during hybrid pipeline: {str(e)}")
        processing_status_fetch["complete"] = "error"
# ------------------- Helpers -------------------
def update_file_registry(filename):
    if not os.path.exists(FILES_TRACK_PATH):
        with open(FILES_TRACK_PATH, 'w'): pass
    with open(FILES_TRACK_PATH, 'r+') as f:
        logger.info(f"Updating file registry for: {filename}")
        lines = [line.strip().split(':')[0] for line in f.readlines()]
        if filename not in lines:
            f.write(f"{filename}:\n")


def sync_file_registry(removed_files):
    if not os.path.exists(FILES_TRACK_PATH):
        return
    with open(FILES_TRACK_PATH, 'r') as f:
        lines = f.readlines()
    with open(FILES_TRACK_PATH, 'w') as f:
        logger.info(f"Syncing registry, removing: {removed_files}")
        for line in lines:
            fname = line.strip().split(":")[0]
            if fname not in removed_files:
                f.write(line)

def load_file_titles():
    titles = []
    try:
        with open(FILES_TRACK_PATH, "r") as file:
            for line in file:
                key, _ = line.strip().split(":")
                titles.append(key)
    except FileNotFoundError:
        pass
    return titles


def load_prompts(file_path):
    prompts = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    prompts[key.strip()] = value.strip()
    return prompts


# ------------------- Main -------------------
if __name__ == '__main__':
    app.run(debug=True)
