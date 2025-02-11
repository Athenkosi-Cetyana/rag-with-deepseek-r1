# RAG with DeepSeek-R1, Ollama, LangChain

## Overview

RAG with DeepSeek-R1 is an intelligent document assistant application that leverages the power of DeepSeek-R1, Ollama, and LangChain to provide concise and factual answers to user queries based on the content of uploaded PDF documents. The application is built using Streamlit for a seamless and interactive user experience.

## Features

- **PDF Document Analysis**: Upload PDF documents for analysis.
- **Intelligent Query Response**: Ask questions about the document and receive concise, factual answers.
- **Document Chunking**: Efficiently processes and chunks documents for better analysis.
- **Vector Store**: Uses an in-memory vector store for document indexing and similarity search.
- **Customizable Models**: Choose between different DeepSeek-R1 models for query processing.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/RAG-withDeepSeek-R1.git
    cd RAG-withDeepSeek-R1
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run rag_deep.py
    ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload a PDF document using the file uploader.

4. Ask questions about the document in the chat input and receive answers from the AI assistant.

## Configuration

### Model Selection

You can select different models for the AI assistant from the sidebar in the Streamlit application. The available models are:
- `deepseek-r1:1.5b`
- `deepseek-r1:3b`

### Customization

You can customize the appearance and behavior of the application by modifying the `rag_deep.py` and `app.py` files.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [Ollama](https://ollama.ai/)

