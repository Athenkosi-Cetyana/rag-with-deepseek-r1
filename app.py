from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import requests

app = Flask(__name__)
CORS(app)

PDF_STORAGE_PATH = 'uploads/pdfs/'
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:latest")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:latest")

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.filename
    with open(file_path, "wb") as file:
        file.write(uploaded_file.read())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    model = request.form.get('model', 'deepseek-r1:latest')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            file_path = save_uploaded_file(file)
            raw_docs = load_pdf_documents(file_path)
            processed_chunks = chunk_documents(raw_docs)
            index_documents(processed_chunks)
            return jsonify({'message': 'File processed successfully', 'file_path': file_path}), 200
        except Exception as e:
            return jsonify({'error': f'Model {model} not available at the moment'}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf_from_server():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    model = request.form.get('model', 'deepseek-r1:latest')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            file_path = save_uploaded_file(file)
            raw_docs = load_pdf_documents(file_path)
            processed_chunks = chunk_documents(raw_docs)
            index_documents(processed_chunks)
            return jsonify({'message': 'File processed successfully', 'file_path': file_path}), 200
        except Exception as e:
            return jsonify({'error': f'Model {model} not available at the moment'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_query = data.get('query')
    model = data.get('model', 'deepseek-r1:latest')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    try:
        relevant_docs = find_related_documents(user_query)
        answer = generate_answer(user_query, relevant_docs)
        return jsonify({'answer': answer}), 200
    except Exception as e:
        return jsonify({'error': f'Model {model} not available at the moment'}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question_from_server():
    data = request.get_json()
    user_query = data.get('query')
    model = data.get('model', 'deepseek-r1:latest')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    try:
        relevant_docs = find_related_documents(user_query)
        answer = generate_answer(user_query, relevant_docs)
        # Send the answer to the other server
        requests.post('http://localhost:3000/receive_answer', json={'answer': answer})
        return jsonify({'answer': answer}), 200
    except Exception as e:
        return jsonify({'error': f'Model {model} not available at the moment'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0')
